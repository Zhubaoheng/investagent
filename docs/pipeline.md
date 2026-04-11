# PoorCharlie Pipeline 文档

## 总览

PoorCharlie 是一个芒格式价值投资多 Agent 系统，通过 10 个阶段的流水线评估一家公司是否值得投资。系统不预测股价，而是判断：公司能否被理解、公开信息是否充分、是否满足质量和估值标准。

## 核心原则

- **风险 = 永久性资本损失**，不是价格波动
- 所有 Agent 必须区分：**事实 (FACT) / 推断 (INFERENCE) / 未知 (UNKNOWN)**
- 任何 Agent 可以输出"停止 / 不知道 / 拒绝继续"
- 默认怀疑复杂性、黑箱、过度叙事、激励扭曲和财务幻觉

## 目标市场

| 市场 | 交易所 | 报告类型 | 会计准则 | 数据源 |
|------|--------|---------|---------|--------|
| A 股 | SSE / SZSE / BSE | 年报、半年报 | CAS | cninfo.com.cn (requests) |
| 港股 | HKEX | Annual Report、Interim Report | IFRS / HKFRS | hkexnews.hk (requests) |
| 美股中概 | NYSE / NASDAQ | 20-F、6-K | US GAAP / IFRS | SEC EDGAR (edgartools) |

---

## Pipeline 流程

```
                        ┌──────────────────────────────────────────────────────┐
                        │                   Soul Prompt (共享)                  │
                        │  芒格式怀疑主义 · 事实/推断/未知 · 永久损失优先         │
                        └──────────────────────────────────────────────────────┘

CompanyIntake ──┐
                │
                ▼
    ┌─────────────────────┐     ┌──────────────────┐
    │  Stage 1: InfoCapture│────→│ 真实数据源         │
    │  信息捕获 (混合 Agent) │◄────│ cninfo / HKEX /  │
    └────────┬────────────┘     │ EDGAR / yfinance  │
             │                  └──────────────────┘
             ▼
    ┌─────────────────────┐     ┌──────────────────┐
    │  Stage 2: Filing    │────→│ 下载 PDF/HTML     │
    │  财报结构化 (混合 Agent)│◄────│ pymupdf4llm 提取  │
    └────────┬────────────┘     │ + LLM 结构化      │
             │                  └──────────────────┘
             ▼
    ┌─────────────────────┐
    │  Stage 3: Triage    │──── REJECT ──→ 🛑 停止
    │  初筛（基于真实数据）   │──── WATCH ───→ 继续
    └────────┬────────────┘──── PASS ────→ 继续
             │
             ▼
    ┌───────────────────────────────────────────────────────┐
    │           Stage 4-8: 并行分析（asyncio.gather × 9）      │
    │                                                       │
    │  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐  │
    │  │Accounting│ │Financial │ │Net Cash │ │Valuation │  │
    │  │  Risk    │ │ Quality  │ │         │ │          │  │
    │  └──────────┘ └──────────┘ └─────────┘ └──────────┘  │
    │                                                       │
    │  ┌───────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │
    │  │ Moat  │ │Compnd│ │Psych │ │System│ │ Ecol │      │
    │  │ 护城河 │ │ 复利  │ │ 心理  │ │ 系统  │ │ 生态  │      │
    │  └───────┘ └──────┘ └──────┘ └──────┘ └──────┘      │
    └────────────────────┬──────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────┐
    │         并行后 Gate 检查（顺序执行）              │
    │                                             │
    │  1. Accounting Risk: RED → 🛑 停止           │
    │  2. Financial Quality: POOR → 🛑 停止        │
    │     （例外：moat=WIDE 或 compounding=STRONG   │
    │      可以覆盖 POOR，允许继续）                  │
    └────────────────────┬────────────────────────┘
                         │
                         ▼
    ┌─────────────────────┐
    │  Stage 9: Critic   │
    │  批评家（纯对抗）      │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐     ┌───────────────────────┐
    │  Stage 10: Committee│────→│ REJECT / TOO_HARD     │
    │  投资委员会            │     │ WATCHLIST / DEEP_DIVE │
    └─────────────────────┘     │ SPECIAL_SIT / INVESTABLE│
                                └───────────────────────┘
```

---

## 各 Agent 详细说明

### Stage 1: Info Capture Agent（信息捕获）

混合 Agent：Phase 1 调用真实 API 获取数据，Phase 2 LLM 补充公司档案，Phase 3 用真实数据覆盖 LLM 输出。

| 项目 | 说明 |
|------|------|
| 输入 | `CompanyIntake` + 真实数据源 API |
| 输出 | `InfoCaptureOutput` |
| 门控 | 无 |

**输出字段**：

| 字段 | 来源 | 说明 |
|------|------|------|
| `company_profile` | LLM | 全称、主营业务、实控人、管理层等 |
| `filing_manifest` | **数据源** | `list[FilingRef]`，每份含类型/年度/URL（7 年回溯）|
| `market_snapshot` | **数据源** | 价格、市值、EV、PE、PB、股息率 |
| `official_sources` | LLM | 官方 IR 页面、交易所公告 |
| `trusted_third_party_sources` | LLM | 研报、评级平台 |
| `missing_items` | LLM | 缺失报告及原因 |

副作用：将 `list[FilingDocument]` 存入 `ctx.data["filing_documents"]` 供下游使用。

---

### Stage 2: Filing Agent（财报结构化）

混合 Agent：下载 PDF → pymupdf4llm 提取 markdown → 关键词分割章节 → LLM 结构化。

| 项目 | 说明 |
|------|------|
| 输入 | `CompanyIntake` + `ctx.data["filing_documents"]` |
| 输出 | `FilingOutput`（最复杂的 schema，230+ 行） |
| 门控 | 无 |

**并行架构**：
- 6 份年报/半年报通过 `asyncio.gather` 并行处理
- PDF 提取（pymupdf4llm）在 `ProcessPoolExecutor` 中运行，绕过 GIL，最多 4 个 worker 真正并行
- 每份 PDF 提取 ~58 秒，6 份并行 ≈ ~90 秒（vs 串行 ~350 秒）
- A 股/港股财务数字由 AkShare API 提供（零幻觉），LLM 仅提取定性段落（会计政策、风险因素等）

**章节提取**（pymupdf4llm → 关键词匹配）：

| 类别 | 提取内容 |
|------|---------|
| 三大报表 | 利润表、资产负债表、现金流量表 |
| 分部数据 | 分业务/地区收入和利润 |
| 会计政策 | 逐类别保留**原文** + 变更标记 |
| 债务结构 | 债务条款、利率、到期日、covenant |
| 非经常损益 | 分类 + 频率 |
| 集中度 | 前 5 客户/供应商占比、地区分布 |
| 资本配置 | 回购/并购/分红历史 |
| 脚注原文 | 债务/租赁/诉讼/关联方 |
| 风险因素 | 分类 + 严重性评级 |

**关键设计**：完整 PDF 文本（~200K 字符）经章节提取后压缩至 ~30K 字符（~6%），控制 LLM 输入 token。下游 Agent 只看结构化 JSON，不看原始文本。

---

### Stage 3: Triage Agent（初筛）

基于 InfoCapture + Filing 的真实数据评估可解释性。

| 项目 | 说明 |
|------|------|
| 输入 | `CompanyIntake` + `InfoCaptureOutput` + `FilingOutput` |
| 输出 | `TriageOutput` |
| 门控 | REJECT → 停止 pipeline |

**四维评分**（1-10 分）：

| 维度 | 含义 |
|------|------|
| `business_model` | 能否从分部数据解释商业模式？ |
| `competition_structure` | 竞争格局能否从数据识别？ |
| `financial_mapping` | 财报是否充足（目标 5 年）？有无不透明结构？ |
| `key_drivers` | 关键驱动因素能否从数据追踪？ |

**自动降分**：零份财报 → 最高 2 分 | < 3 年数据 → 最高 5 分 | 政策变更 → 扣分

**决策**：PASS (均分 ≥ 7) / WATCH (5-7) / REJECT (< 5)

---

### Stage 4-8: 并行分析（9 Agent 同时运行）

Triage 通过后，以下 9 个 Agent 通过 `asyncio.gather()` **同时运行**，互不依赖，均依赖 Filing 输出：

#### Accounting Risk Agent（会计风险）

| 项目 | 说明 |
|------|------|
| 输入 | `CompanyIntake` + `FilingOutput`（完整 JSON 注入 prompt） |
| 输出 | `AccountingRiskOutput` |
| 门控 | RED → 并行完成后停止 pipeline |

**10 项检查**：收入确认变更、合并范围变更、分部披露变更、折旧政策变更、存货计价变更、坏账计提变更、一次性项目正常化、Non-GAAP 激进度、审计意见变化、财务重述

**风险等级**：GREEN / YELLOW / RED

#### Financial Quality Agent（财务质量）

| 项目 | 说明 |
|------|------|
| 输入 | `CompanyIntake` + `FilingOutput` |
| 输出 | `FinancialQualityOutput` |
| 门控 | POOR → 并行完成后停止（但 moat=WIDE 或 compounding=STRONG 可覆盖） |

**六维评分**（1-10 分）：

| 维度 | 核心指标 |
|------|---------|
| `per_share_growth` | EPS/FCF 增长，摊薄影响 |
| `return_on_capital` | ROIC/ROE/ROA |
| `cash_conversion` | CFO/NI, FCF/NI |
| `leverage_safety` | 净债务/EBIT，利息覆盖 |
| `capital_allocation` | 回购/分红/并购质量 |
| `moat_financial_trace` | 毛利率/营业利润率稳定性 |

**通过标准**：均分 ≥ 5 且无单项 ≤ 2

#### Net Cash Agent（净现金）

| 项目 | 说明 |
|------|------|
| 输入 | `CompanyIntake` + `FilingOutput` |
| 输出 | `NetCashOutput` |

**核心计算**：净现金 = 现金 + 短期投资 − 有息负债

**关注级别**：NORMAL (≤ 0.5x) / WATCH (0.5-1.0x) / PRIORITY (1.0-1.5x) / HIGH_PRIORITY (> 1.5x)

#### Valuation Agent（估值）

| 项目 | 说明 |
|------|------|
| 输入 | `CompanyIntake` + `FilingOutput` + `MarketSnapshot` |
| 输出 | `ValuationOutput` |

**三情景**：Bear / Base / Bull → 穿透回报 → 摩擦调整后回报 → 对比门槛利率

**确定性后处理**：LLM 输出 per-method IV 估计 → Python 取中位数 → 计算 MoS/price_vs_value → 异常 IV 过滤（排除 < 5% 或 > 20x 当前价格的估计） → MoS 限制在 [-100%, 100%]

#### Mental Models（心智模型 × 5）

五个 Agent 与上述四个 Agent **同时运行**，每个都接收完整 `FilingOutput` JSON：

| Agent | 核心问题 | 关键输出 |
|-------|---------|---------|
| **Moat** | 竞争优势是什么？ | 行业结构、护城河类型、定价权、趋势 |
| **Compounding** | 复利引擎是否健全？ | ROIC、增量回报、可持续期、每股增长逻辑 |
| **Psychology** | 有哪些行为偏差？ | 管理层激励扭曲、市场情绪偏差、叙事与事实背离 |
| **Systems** | 单点故障在哪？ | 单点故障、脆弱源、容错能力、系统韧性评级 |
| **Ecology** | 生态位和适应力？ | 生态位、适应性趋势、周期 vs 结构、10-20 年存活率 |

---

### Stage 9: Critic Agent（批评家）

| 项目 | 说明 |
|------|------|
| 输入 | `FilingOutput` + 全部上游 Agent 输出 JSON |
| 输出 | `CriticOutput` |

**规则**：永远不复述多头论点。纯对抗性。

| 字段 | 问题 |
|------|------|
| `kill_shots` | 公司会死在哪里？ |
| `permanent_loss_risks` | 什么导致永久资本损失？ |
| `moat_destruction_paths` | 什么摧毁护城河？ |
| `management_failure_modes` | 管理层如何毁灭价值？ |
| `what_would_make_this_uninvestable` | 什么条件下不可投资？ |

---

### Stage 10: Investment Committee Agent（投资委员会）

| 项目 | 说明 |
|------|------|
| 输入 | **全部** 13 个上游 Agent 输出（JSON 汇总注入 prompt） |
| 输出 | `CommitteeOutput` |

**规则**：不重新分析原始数据，只综合上游结论。

**六个最终标签**：

| 标签 | 含义 |
|------|------|
| `REJECT` | 不投资（门控失败 / kill shot / 回报不达标） |
| `TOO_HARD` | 太难判断（关键未知不可解决） |
| `WATCHLIST` | 观察（有潜力但时机或信息不足） |
| `DEEP_DIVE` | 深入研究（基本面吸引但需更多细节） |
| `SPECIAL_SITUATION` | 特殊情况（困境反转、重组、分拆） |
| `INVESTABLE` | 可投资（通过所有门控 + 回报 ≥ 10%） |

**输出字段**：thesis / anti_thesis / largest_unknowns / expected_return_summary / why_now_or_why_not_now / next_action

---

## 数据流向图

```
CompanyIntake
    │
    ├─→ [1] InfoCapture ─┬──→ company_profile, filing_manifest, market_snapshot
    │    (串行)           └──→ ctx.data["filing_documents"] (原始 PDF/HTML)
    │
    ├─→ [2] Filing ──────────→ income_statement, balance_sheet, cash_flow,
    │    (串行, PDF下载+提取)    segments, accounting_policies, debt_schedule,
    │                          special_items, concentration, footnotes, risks
    │
    ├─→ [3] Triage ──────────→ decision, scores, fatal_unknowns
    │    (串行, REJECT 则停止)
    │
    │    ┌─ asyncio.gather ──────────────────────────────────────────────┐
    │    │                                                              │
    ├─→  │ [4] AccountingRisk ──→ risk_level, major_changes, credibility│
    ├─→  │ [5] FinancialQuality → 6 scores, pass/fail, quality         │
    ├─→  │ [6] NetCash ─────────→ net_cash, ratio, attention_level     │
    ├─→  │ [7] Valuation ───────→ scenario_returns, meets_hurdle       │
    ├─→  │ [8] Moat ────────────→ industry_structure, moat_type        │
    ├─→  │ [8] Compounding ─────→ engine, incremental_roic             │
    ├─→  │ [8] Psychology ──────→ incentive_distortion, sentiment      │
    ├─→  │ [8] Systems ─────────→ single_points, fragility, resilience │
    ├─→  │ [8] Ecology ─────────→ niche, adaptability, survival_prob   │
    │    │    (以上 9 个并行，均依赖 FilingOutput JSON，互不依赖)          │
    │    └──────────────────────────────────────────────────────────────┘
    │         │
    │         ├── Gate: AccountingRisk RED → 🛑 停止
    │         ├── Gate: FinancialQuality POOR → 🛑 停止
    │         │    （覆盖条件：moat=WIDE 或 compounding=STRONG → 继续）
    │         ▼
    ├─→ [9] Critic ──────────→ kill_shots, permanent_loss, moat_destruction
    │    (串行, 接收全部上游 Agent 输出 JSON)
    │
    └─→ [10] Committee ──────→ final_label, thesis, anti_thesis, next_action
         (串行, 接收全部 13 个上游 Agent 输出 JSON)
```

---

## 输出文件

每次运行生成两个文件：

| 文件 | 格式 | 用途 |
|------|------|------|
| `{ticker}_{timestamp}.md` | Markdown | 人类可读报告（概览表 + 每 Stage 详细输出） |
| `{ticker}_{timestamp}_debug.json` | JSON | 完整 Agent 输入输出日志（零截断，用于 debug） |

---

## 实现状态

| 组件 | 状态 | 说明 |
|------|------|------|
| Soul Prompt | ✅ | 所有 Agent 共享，回测模式注入截止日期约束（禁用未来信息） |
| 14 个 Agent | ✅ | 全部接入真实数据，带 retry + JSON 修复 |
| 数据源层 | ✅ | EDGAR / cninfo / HKEX / yfinance，7 年回溯 |
| InfoCapture 混合 Agent | ✅ | 真实 fetcher + LLM，数据覆盖 LLM 输出 |
| Filing 混合 Agent | ✅ | PDF 下载 → pymupdf4llm（多进程并行） → 章节提取 → LLM 结构化 |
| Triage 后置 | ✅ | 基于真实 Filing 数据评估可解释性 |
| 下游 context 注入 | ✅ | 全部 11 个下游 Agent 注入 FilingOutput JSON |
| Critic/Committee 综合 | ✅ | 接收全部上游 Agent 输出汇总 |
| CLI | ✅ | `poorcharlie 1448.HK` 一行命令 |
| 报告生成 | ✅ | Markdown + JSON debug log |
| 212 tests | ✅ | 全部通过 |
| 端到端验证 | ✅ | 福寿园 1448.HK：14 Agent，157K tokens，907s → REJECT |
| Valuation 后处理 | ✅ | 确定性 MoS 计算 + 异常 IV 过滤 + clamp [-100%, 100%] |
| 回测模式 | ✅ | `as_of_date` 截止日期约束，禁止使用未来信息/联网搜索/训练数据泄露 |
| PDF 多进程提取 | ✅ | `ProcessPoolExecutor` 绕过 GIL，6 份 PDF 真正并行 |
