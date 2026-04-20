# Spec: 预筛选 + 组合构建（永久功能）

扩展 poorcharlie pipeline，新增两个能力：全市场预筛选、组合级持仓管理。这些是产品的永久功能，不依赖回测。

## 1. 股票池构建

### 1.1 股票宇宙

给定目标市场（当前仅 A 股），获取该时点的全部正股列表。

### 1.2 排除规则

排除分两层：能用规则判定的用规则，需要理解上下文的交给 LLM。

**规则排除**（纯 Python，零成本）：

- ST / \*ST：股票名称前缀检测
- 财务信息披露不超过 3 年：可获取的年报数量
- 金融类（银行/保险/券商）：申万一级行业代码（银行、非银金融）

**LLM 排除**（需要判断力，但不涉及投资决策）：

- 创立时间不足 5 年：部分公司注册日期不直接可得，LLM 根据公司基本信息判断
- 不透明科技（军工、尖端材料、创新药）：行业边界模糊，靠行业代码无法精确覆盖（如某公司主业是民用但有军工子业务），LLM 根据主营业务描述判断
- 壳公司 / 资产极度空洞：没有统一的量化阈值，LLM 综合营收、资产、员工数等信息判断

LLM 排除输入：公司名称、行业分类、主营业务描述、上市日期、基本财务概况（来自 AkShare）。输出：`EXCLUDE`（附理由）或 `KEEP`。

排除后有效池预计约 3000~3500 支。

## 2. 财务比率计算

纯 Python 模块，从 AkShare 三表数据计算关键比率，作为筛选 agent 的输入。不做任何排序或判断。

### 2.1 计算指标

- **盈利能力**：ROE、ROIC、毛利率、净利率
- **成长性**：营收增速（YoY）、净利润增速（YoY）、EPS 增速
- **现金质量**：经营现金流/净利润、自由现金流/净利润、Capex/营收
- **杠杆**：资产负债率、净负债/EBIT、利息覆盖倍数
- **估值快照**：PE（滚动）、PB、股息率（如可获取）

输出为近 3~5 年序列，每年一组比率值。

### 2.2 输入/输出

- 输入：AkShare 三表原始数据（IncomeStatementRow、BalanceSheetRow、CashFlowRow）
- 输出：`dict[str, list[float | None]]`，按指标名索引，每个值对应一个年度

## 3. 预筛选 Agent

### 3.1 定位

pipeline 的最前端环节，在 Filing Structuring 之前。对排除规则通过后的全部公司做轻量判断，过滤明显不值得深入分析的公司。

**目标**：宁可放进来 100 家平庸公司，也不能漏掉 1 家好公司。

### 3.2 输入

全部由 Python 预计算，不使用 LLM 生成：

1. 公司基本信息：行业分类、主营业务描述、上市时间、市值（来源：AkShare）
2. 财务比率序列：§2 计算的全部比率（近 3~5 年）
3. 格式化为一段结构化文本

不同行业的"好"标准不同（消费品公司 15% ROE vs 重工业公司 15% ROE 含义完全不同），需要 LLM 理解行业上下文后判断。

### 3.3 输出 Schema

```python
class ScreenerOutput(BaseAgentOutput):
    decision: str        # "SKIP" | "PROCEED" | "SPECIAL_CASE"
    reason: str          # 简短理由
    industry_context: str  # LLM 对该公司行业特征的简要说明
```

### 3.4 约束

- **不使用硬指标阈值**。LLM 需理解上下文——某家优秀公司某年遇到特殊情况导致指标异常，不应被硬卡掉。
- Prompt 要求 LLM 输出简洁，控制思考链长度（这是轻量筛选，不是深度分析）。

## 4. 组合构建 Agent

### 4.1 定位

pipeline 终端环节。在多家公司各自跑完 pipeline 后，从所有 INVESTABLE 标的中选股并分配仓位。

### 4.2 输入

- 所有结论为 `INVESTABLE` 的标的列表
- 每个标的的：`enterprise_quality`、`price_vs_value`、`margin_of_safety_pct`、`meets_hurdle_rate`、行业分类
- 当前持仓状态（如有）
- 可用资金

### 4.3 输出 Schema

```python
class PortfolioAllocation(BaseModel, frozen=True):
    ticker: str
    target_weight: float   # 0.05 ~ 0.30
    reason: str

class PortfolioOutput(BaseAgentOutput):
    allocations: list[PortfolioAllocation]
    cash_weight: float     # 剩余现金比例
    industry_distribution: dict[str, float]  # 行业 → 合计权重
    rebalance_actions: list[str]  # 本次调仓动作描述
```

### 4.4 选股优先级

遵循芒格核心原则：

1. `enterprise_quality = GREAT` 且 `price_vs_value = FAIR` 或更好
2. `enterprise_quality = GREAT` 且 `price_vs_value = CHEAP`
3. `enterprise_quality = AVERAGE` 且 `price_vs_value = CHEAP`

不投资 `POOR` 企业，无论估值多低。

### 4.5 持仓约束

- 最多 10 个持仓，不硬凑——如果只有 3 个好主意，就持 3 个 + 现金
- 单只仓位下限 5%，上限 30%
- 兼顾行业分散，避免单一行业过度集中

### 4.6 卖出决策

卖出决策由组合构建 agent 统一做出（非独立的卖出规则），确保卖出和买入在同一个全局视角下决策。

触发条件（优先级从高到低）：

1. **基本面严重恶化**：pipeline 结论从 `INVESTABLE` 降级为 `REJECT` / `TOO_HARD`
2. **估值严重过高**：`price_vs_value` 变为 `EXPENSIVE`，安全边际为负且超过阈值
3. **出现明显更好的机会**：候选标的性价比显著优于当前持仓

### 4.7 Change-Triggered Rebalancing（变化触发的调仓）

**问题**：扫描日不应该自动等于调仓日。芒格原则——*"Never interrupt compound interest unnecessarily."* 如果没有实质变化，组合应原样保留。

**设计**：在调用 PortfolioStrategy LLM **之前**，由规则引擎（`workflow/change_detector.py`）检查是否存在任何"值得重新决策"的触发。若没有任何触发，直接返回上期组合权重，**完全跳过 LLM 调用**。

**5 个触发条件**（任一命中即唤醒 LLM）：

| 编号 | 触发名 | 检测逻辑 | 预期动作 |
|------|--------|---------|---------|
| T1 | `QUALITY_DOWNGRADE` | 任一已持仓标的 `enterprise_quality` 相比上期降级（按 GREAT > GOOD > AVERAGE > BELOW_AVERAGE > POOR 排序） | 考虑 REDUCE 或 EXIT |
| T2 | `NEW_KILL_SHOT` | 任一已持仓标的 Critic 输出的 `kill_shots` 集合出现了**新增条目**（相比上期） | 严肃考虑 EXIT |
| T3 | `ACCOUNTING_RED` | 任一已持仓标的的 `accounting_risk_level` 变为 `RED`（上期不是 RED） | 立即 EXIT |
| T4 | `NEW_INVESTABLE` | 存在未持仓的标的 `final_label=INVESTABLE` 且上期不是 INVESTABLE | 评估是否建仓（可能需要减弱仓腾地方） |
| T5 | `LABEL_REJECT` | 任一已持仓标的 `final_label` 降为 `REJECT` / `TOO_HARD` | EXIT |

**冷启动例外**：若 `CandidateStore` 中没有任何候选的 `prev_scan_date`（即首次扫描），返回一个合成的 cold-start trigger，让 LLM 正常运行建仓逻辑。

**持久化**：`CandidateSnapshot` 新增字段用于变化检测：
- `kill_shots: list[str]` — 当期 Critic 输出
- `accounting_risk_level: str` — 当期 AccountingRisk 输出（GREEN/YELLOW/RED）
- `prev_final_label`, `prev_enterprise_quality`, `prev_kill_shots`, `prev_accounting_risk_level`, `prev_scan_date` — 上期值，由 `ingest_scan_results` 在覆盖前保留

**LLM 端的配合**：当 change_detector 发现触发并唤醒 LLM 时，`PortfolioStrategyInput.change_triggers` 把触发列表传给 LLM，prompt 明确要求：
> "默认动作是**维持上期组合不变**。只对 change_triggers 里明确列出的标的做调整。没有触发的持仓一律 HOLD 且保留上期 target_weight。"

这条硬约束和 LLM 的默认"全面重估"倾向作对，确保即便 LLM 被唤醒，它也只动该动的部分。

**结果**：大多数扫描日将发现"无触发" → 零 LLM 调用 → 零交易摩擦 → 复利不被打断。仅当市场/基本面发生实质变化时，系统才动作。

### 4.8 持仓分级：CORE vs SATELLITE

**问题**：当前系统对所有持仓用同一套卖出规则。但茅台和东航物流本质不同——前者是芒格说的"复利机器"，后者是周期机会。两者应该被**差异化保护**。

**设计**：`PositionDecision` 新增 `position_tier: PositionTier`（`CORE` 或 `SATELLITE`），`PortfolioHolding` 同步持久化 tier。

**CORE 的三条必要条件**（同时满足）：
1. `enterprise_quality == GREAT`
2. `final_label == INVESTABLE`
3. 投资论点（thesis/strengths）明确描述护城河宽深、可持续、每年变宽——不是"还不错"而是"10年后依然稳固"

**SATELLITE**：其他所有 `INVESTABLE / DEEP_DIVE / SPECIAL_SITUATION` 中不满足上述三条的标的。周期股、转机股、护城河中等的好生意、估值驱动的机会。

**差异化卖出规则**：

| 卖出原因 | CORE | SATELLITE |
|---------|:------:|:-----------:|
| 估值过贵（MoS 负、PE 高位） | **不卖** | 允许 REDUCE |
| Committee label → WATCHLIST | **不卖** | 允许 REDUCE |
| 出现更好的相对机会 | **不卖** | 允许轮换 |
| quality → GOOD / AVERAGE | REDUCE 仓位但不 EXIT | REDUCE 或 EXIT |
| quality → BELOW_AVERAGE / POOR | 允许 EXIT | EXIT |
| Critic 非空 kill_shots | 允许 EXIT | EXIT |
| AccountingRisk = RED | 立即 EXIT | 立即 EXIT |
| label → REJECT / TOO_HARD | 允许 EXIT | EXIT |

核心原则：**CORE 仓只因"公司永久性受损"卖出，绝不因"价格看起来贵"卖出。**

**Post-process 防呆**（`agents/portfolio_strategy.py::_enforce_core_tier_rules`）：
如果 LLM 把 `enterprise_quality != GREAT` 或 `final_label != INVESTABLE` 的标的标为 CORE，规则引擎自动降级为 SATELLITE。这防止 LLM 因乐观情绪滥用 CORE 分类。规则只降级不升级——升级完全是 LLM 的判断。

**change_detector 的 tier 感知**：当持仓触发变化（QUALITY_DOWNGRADE / NEW_KILL_SHOT 等），trigger 的 `detail` 字段会带上 `[CORE]` 或 `[SATELLITE]` 前缀，传给 PortfolioStrategy LLM 和未来的 Trader Agent 使用。

**为什么重要**：上一次回测的最大败笔是卖掉茅台（GREAT+INVESTABLE+WIDE_MOAT）去买格力、济川——这是典型的 CORE → SATELLITE 误配置。分级后，茅台这类标的只能在基本面永久恶化时才被卖出，避免"估值驱动的轮换"侵蚀复利。

### 4.9 Trader Agent — Layer 4 执行裁决者

**问题**：到上一步为止，PortfolioStrategy 给出的是"目标组合 + 分级"。但还缺一层——**这些决策应该怎么执行？什么节奏？当下市场环境是否适合立即动作？** 把投研职责和执行职责合在一个 LLM 里，会让基金经理替交易员操心，同时让交易员替基金经理做决策。这不符合机构投资的职责划分。

**设计**：新增 Trader Agent 作为 Layer 4，位于 PortfolioStrategy 之后。采用**三层三明治架构**，和 Committee Agent 同样的 LLM + 规则混合模式：

```
PortfolioStrategy 输出 (proposed_decisions)
  ↓
Trader.pre_check（规则，零成本）
  - CORE 仓 EXIT/REDUCE 硬阻断（若无 kill_shot / RED / 质量崩塌 / label 降级）
  - 单一仓位 > 20% 截断
  - 总权重 > 100% 归一化
  - 被阻断的订单: action = BLOCKED, target_weight 回到当前持仓权重
  ↓
Trader.LLM（市场判断）
  - market_regime: PANIC / NORMAL / EUPHORIA / UNKNOWN
  - 每条订单打 urgency: IMMEDIATE / NORMAL / PATIENT
  - market_assessment: 一句话市场环境概述
  - execution_plan_summary: 整体执行计划
  - **LLM 不能改变规则层设定的 action**
  ↓
Trader.post_check（规则兜底）
  - 如果 LLM 试图把 BLOCKED 改回 EXIT → 强制改回 BLOCKED
  - 如果 LLM 超仓位上限 → 截断
  - 如果 LLM 漏了订单 → 从 pre_check 结果重建
  - 如果 LLM 幻觉了新订单 → 丢弃
  ↓
TraderOutput（最终订单簿）
```

**规则层保护的边界（pre_check + post_check 双重防护）**：

| 情形 | 规则层动作 |
|------|-----------|
| CORE 持仓 action=EXIT 且候选无 kill_shot/无 RED/quality 仍 GREAT/label 仍 INVESTABLE | action → BLOCKED, weight 保留原值 |
| CORE 持仓 action=REDUCE 同上条件 | action → BLOCKED, weight 保留原值 |
| CORE 持仓 action=EXIT 且 quality 降为 POOR | 允许 EXIT |
| CORE 持仓 action=EXIT 且 Critic 给出 kill_shot | 允许 EXIT |
| CORE 持仓 AccountingRisk=RED | 允许 EXIT |
| SATELLITE 持仓任何卖出 | 允许 EXIT |
| target_weight > 20% | 截断为 20% |
| 总非阻断权重 > 100% | 按比例归一化 |

**LLM 的职责边界**：

- 能做：评估市场环境、为订单分配紧急度、写执行说明
- 不能做：推翻规则层的 BLOCKED、修改 action、增减订单条目
- 默认倾向：保守执行、减少冲击、尊重上游决策

**紧急度语义**（Urgency → 执行天数映射，由下游 executor 消费）：

| Urgency | 场景 | 天数 |
|---------|------|------|
| IMMEDIATE | RED 会计风险、kill_shot 驱动的 EXIT、PANIC 中的 CORE 加仓 | 1-2 |
| NORMAL | 常规调仓 | 5 |
| PATIENT | EUPHORIA 中的 BUY、CORE 的非紧急 ADD | 10+ |

**与既有 MungerStrategy 执行器的关系**：
当前回测执行器仅读 target_weight，对 urgency 标签尚未消费。Trader 的 tier/urgency 元数据会被持久化到 `candidate_store.json` 中的 `PortfolioHolding.position_tier` 字段（tier 已落地，urgency 留到后续版本消费）。

**失败回退**：
- Trader LLM 失败 → TraderAgent 内部回退到"规则-only"输出（orders 来自 pre_check 结果）
- TraderAgent 整体失败 → decision_pipeline 回退使用 PortfolioStrategy 原始输出

**关键效果**：上次回测中茅台在 2024-05（-15%）和 2025-09（-30%）被系统卖出的操作，在新架构下会被 pre_check 阻断——除非 Critic 在这些时点发现了茅台的 kill_shot 或会计转红。**CORE 仓的生命周期由"基本面永久性受损"决定，而不是"估值暂时难看"**。

## 5. 多 LLM Provider 支持

系统需支持多个 LLM provider：

- 排除规则判断可使用免费/低成本模型（如 MiniMax）
- 筛选 + 完整 pipeline + 组合构建使用主力模型（当前为 DeepSeek R1）

通过配置文件或环境变量切换 provider 和 API key，不硬编码到业务逻辑中。现有 `llm.py` 需扩展以支持多 provider 路由。

## 6. 实现模块

以下模块放在 `src/poorcharlie/` 下，正常写测试：

- `screening/universe.py` — 股票池构建 + 排除规则
- `screening/ratio_calc.py` — 财务比率计算（纯 Python）
- `screening/screener.py` — 预筛选 agent 实现
- `agents/portfolio.py` — 组合构建 agent 实现
- `prompts/templates/screener.txt` — 预筛选 prompt
- `prompts/templates/portfolio.txt` — 组合构建 prompt
- `schemas/screener.py` — ScreenerOutput
- `schemas/portfolio.py` — PortfolioOutput

## 7. 实现顺序

1. `ratio_calc.py` + 测试（纯计算，无外部依赖）
2. `universe.py` + 测试（AkShare 数据获取 + 规则排除 + LLM 排除）
3. `screener.py` + prompt + schema + 测试
4. `portfolio.py` + prompt + schema + 测试
5. 多 LLM provider 路由（扩展 `llm.py`）
