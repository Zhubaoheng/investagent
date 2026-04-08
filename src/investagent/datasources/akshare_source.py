"""AkShare structured financial data source.

Provides deterministic (zero-hallucination) financial statement data
for A-shares and HK stocks via AkShare aggregated APIs (Sina/东财/同花顺).

This replaces LLM-based number extraction from PDF for standard
three-statement financials, EPS, and shares.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import requests.exceptions
import urllib3.exceptions

logger = logging.getLogger(__name__)

# Retry config for transient AkShare errors (SSL EOF, proxy disconnect, etc.)
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 5  # seconds; actual delay = base * 2^attempt (5, 10, 20)
_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ProxyError,
    requests.exceptions.SSLError,
    requests.exceptions.Timeout,
    urllib3.exceptions.MaxRetryError,
    urllib3.exceptions.NewConnectionError,
    ConnectionResetError,
    OSError,
)

# Global semaphore: py_mini_racer (V8 engine used by AkShare/同花顺) is NOT
# thread-safe. Concurrent asyncio.to_thread() calls crash with
# "FATAL:address_pool_manager.cc Check failed: !pool->IsInitialized()".
# Serialize all AkShare calls to prevent V8 multi-thread initialization.
_AKSHARE_LOCK = asyncio.Semaphore(1)

# Circuit breaker: per-ticker tracking. After N consecutive tickers fail
# (post-retry), pause for a cooldown period then retry. NOT global disable.
_CONSECUTIVE_FAILURES = 0
_CIRCUIT_BREAKER_THRESHOLD = 5
_CIRCUIT_COOLDOWN_UNTIL = 0.0  # timestamp; skip calls until this time
_CIRCUIT_COOLDOWN_SECS = 300  # 5 minutes cooldown, then retry


def _akshare_call_with_retry(func, *args, label: str = ""):
    """Call an AkShare API function with exponential backoff retry on transient errors.

    Proxy decisions are handled by ClashRotator.patch_requests() — direct-first,
    proxy only after consecutive failures. We just retry here, no proxy manipulation.
    """
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            return func(*args)
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "AkShare %s transient error (attempt %d/%d), retrying in %ds: %s",
                label, attempt + 1, _MAX_RETRIES, delay, type(exc).__name__,
            )
            time.sleep(delay)
        except Exception:
            raise  # non-retryable (e.g., KeyError, ValueError) — propagate immediately
    # All retries exhausted
    raise last_exc  # type: ignore[misc]


def _parse_cn_number(val: Any) -> float | None:
    """Parse Chinese financial number strings like '893.35亿', '137.89亿', '68.64'."""
    if val is None or val is False or (isinstance(val, float) and val != val):  # NaN check
        return None
    s = str(val).strip()
    if not s or s == "False" or s == "nan" or s == "-":
        return None

    # Remove commas
    s = s.replace(",", "")

    # Chinese unit multipliers
    multiplier = 1.0
    if s.endswith("亿"):
        multiplier = 1e8
        s = s[:-1]
    elif s.endswith("万"):
        multiplier = 1e4
        s = s[:-1]
    elif s.endswith("千") or s.endswith("仟"):
        multiplier = 1e3
        s = s[:-1]

    try:
        return float(s) * multiplier
    except (ValueError, TypeError):
        return None


def fetch_a_share_financials(symbol: str, years: int = 7) -> dict[str, Any]:
    """Fetch A-share financial statements from AkShare (Sina source).

    Uses stock_financial_report_sina (新浪财经) instead of THS (同花顺).
    Sina uses pure requests+BeautifulSoup — NO V8/py_mini_racer dependency,
    so this function is thread-safe and can run concurrently without Semaphore.

    Args:
        symbol: Stock code without prefix (e.g., "600519")
        years: Number of annual reports to fetch

    Returns dict with keys: income_statement, balance_sheet, cash_flow,
    each containing a list of row dicts ready for FilingOutput schema.
    """
    import akshare as ak
    import math

    code = re.sub(r"[^\d]", "", symbol.split(".")[0])
    prefix = "sh" if code.startswith(("6", "9")) else "sz"
    sina_code = f"{prefix}{code}"

    def _val(row: Any, col: str) -> float | None:
        v = row.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    result: dict[str, Any] = {
        "income_statement": [],
        "balance_sheet": [],
        "cash_flow": [],
        "source": "akshare_sina",
    }

    # --- Income Statement ---
    try:
        df = _akshare_call_with_retry(
            ak.stock_financial_report_sina, sina_code, "利润表",
            label=f"A-share IS {code}",
        )
        annual = df[df["报告日"].str.endswith("1231")].head(years)

        for _, row in annual.iterrows():
            fiscal_year = str(row["报告日"])[:4]
            result["income_statement"].append({
                "fiscal_year": fiscal_year,
                "fiscal_period": "FY",
                "revenue": _val(row, "营业收入"),
                "cost_of_revenue": _val(row, "营业成本"),
                "gross_profit": None,
                "rd_expense": _val(row, "研发费用"),
                "sga_expense": _val(row, "销售费用"),
                "depreciation_amortization": None,
                "operating_income": _val(row, "营业利润"),
                "interest_expense": _val(row, "利息支出"),
                "tax_provision": _val(row, "所得税费用"),
                "net_income": _val(row, "净利润"),
                "net_income_to_parent": _val(row, "归属于母公司所有者的净利润"),
                "eps_basic": _val(row, "基本每股收益"),
                "eps_diluted": _val(row, "稀释每股收益"),
                "shares_basic": None,
                "shares_diluted": None,
            })
        logger.info("AkShare A-share IS (Sina): %d rows for %s", len(result["income_statement"]), code)
    except Exception:
        logger.warning("AkShare A-share IS failed for %s", code, exc_info=True)

    # --- Balance Sheet ---
    try:
        df = _akshare_call_with_retry(
            ak.stock_financial_report_sina, sina_code, "资产负债表",
            label=f"A-share BS {code}",
        )
        annual = df[df["报告日"].str.endswith("1231")].head(years)

        for _, row in annual.iterrows():
            fiscal_year = str(row["报告日"])[:4]
            result["balance_sheet"].append({
                "fiscal_year": fiscal_year,
                "cash_and_equivalents": _val(row, "货币资金"),
                "short_term_investments": _val(row, "交易性金融资产"),
                "accounts_receivable": _val(row, "应收账款"),
                "inventory": _val(row, "存货"),
                "total_current_assets": _val(row, "流动资产合计"),
                "ppe_net": _val(row, "固定资产及清理合计") or _val(row, "固定资产净额"),
                "goodwill": _val(row, "商誉"),
                "intangible_assets": _val(row, "无形资产"),
                "total_assets": _val(row, "资产总计"),
                "accounts_payable": _val(row, "应付账款"),
                "short_term_debt": _val(row, "短期借款"),
                "total_current_liabilities": _val(row, "流动负债合计"),
                "long_term_debt": _val(row, "长期借款"),
                "total_liabilities": _val(row, "负债合计"),
                "shareholders_equity": _val(row, "归属于母公司股东权益合计"),
                "minority_interest": _val(row, "少数股东权益"),
            })
        logger.info("AkShare A-share BS (Sina): %d rows for %s", len(result["balance_sheet"]), code)
    except Exception:
        logger.warning("AkShare A-share BS failed for %s", code, exc_info=True)

    # --- Cash Flow ---
    try:
        df = _akshare_call_with_retry(
            ak.stock_financial_report_sina, sina_code, "现金流量表",
            label=f"A-share CF {code}",
        )
        annual = df[df["报告日"].str.endswith("1231")].head(years)

        for _, row in annual.iterrows():
            fiscal_year = str(row["报告日"])[:4]
            ocf = _val(row, "经营活动产生的现金流量净额")
            capex = _val(row, "购建固定资产、无形资产和其他长期资产所支付的现金")
            div = _val(row, "分配股利、利润或偿付利息所支付的现金")

            result["cash_flow"].append({
                "fiscal_year": fiscal_year,
                "operating_cash_flow": ocf,
                "capex": abs(capex) if capex else None,
                "free_cash_flow": (ocf - abs(capex)) if ocf is not None and capex is not None else None,
                "dividends_paid": div,
                "buyback_amount": None,
                "debt_issued": _val(row, "取得借款收到的现金"),
                "debt_repaid": _val(row, "偿还债务支付的现金"),
                "acquisitions": None,
            })
        logger.info("AkShare A-share CF (Sina): %d rows for %s", len(result["cash_flow"]), code)
    except Exception:
        logger.warning("AkShare A-share CF failed for %s", code, exc_info=True)

    return result


def _pivot_hk_long_format(df: Any, years: int) -> dict[str, dict[str, float | None]]:
    """Pivot AkShare HK long-format data to {report_date: {item_name: amount}}.

    AkShare returns HK data as long format: one row per (report_date, item).
    We pivot to one dict per report_date for easy mapping.
    """
    result: dict[str, dict[str, float | None]] = {}
    for _, row in df.iterrows():
        report_date = str(row.get("REPORT_DATE", ""))[:10]
        fiscal_year = report_date[:4]
        item_name = str(row.get("STD_ITEM_NAME", ""))
        amount = row.get("AMOUNT")

        if fiscal_year not in result:
            result[fiscal_year] = {}
        if amount is not None and amount == amount:  # NaN check
            result[fiscal_year][item_name] = float(amount)

    # Keep only latest N years
    sorted_years = sorted(result.keys(), reverse=True)[:years]
    return {y: result[y] for y in sorted_years}


# HK item name → our schema field mapping
_HK_IS_MAP: dict[str, str] = {
    "营业额": "revenue",
    "营运收入": "revenue",
    "销售成本": "cost_of_revenue",
    "毛利": "gross_profit",
    "研发费用": "rd_expense",
    "销售及分销费用": "sga_expense",
    "销售费用": "sga_expense",
    "折旧及摊销": "depreciation_amortization",
    "经营溢利": "operating_income",
    "营运利润": "operating_income",
    "融资成本": "interest_expense",
    "税项": "tax_provision",
    "所得税": "tax_provision",
    "除税后溢利": "net_income",
    "净利润": "net_income",
    "持续经营业务税后利润": "net_income",
    "股东应占溢利": "net_income_to_parent",
    "公司拥有人应占利润": "net_income_to_parent",
    "每股基本盈利": "eps_basic",
    "基本每股收益": "eps_basic",
    "每股摊薄盈利": "eps_diluted",
    "稀释每股收益": "eps_diluted",
}

_HK_BS_MAP: dict[str, str] = {
    "物业厂房及设备": "ppe_net",
    "物业、厂房及设备": "ppe_net",
    "商誉": "goodwill",
    "无形资产": "intangible_assets",
    "存货": "inventory",
    "应收帐款": "accounts_receivable",
    "应收账款": "accounts_receivable",
    "现金及等价物": "cash_and_equivalents",
    "银行结余及现金": "cash_and_equivalents",
    "流动资产合计": "total_current_assets",
    "非流动资产合计": "total_current_assets",  # fallback
    "总资产": "total_assets",
    "资产总值": "total_assets",
    "资产总额": "total_assets",
    "应付帐款": "accounts_payable",
    "应付账款": "accounts_payable",
    "短期借款": "short_term_debt",
    "短期贷款": "short_term_debt",
    "流动负债合计": "total_current_liabilities",
    "长期贷款": "long_term_debt",
    "非流动负债合计": "long_term_debt",
    "总负债": "total_liabilities",
    "负债总值": "total_liabilities",
    "负债总额": "total_liabilities",
    "股东权益": "shareholders_equity",
    "权益总额": "shareholders_equity",
    "净资产": "shareholders_equity",
    "少数股东权益": "minority_interest",
}

_HK_CF_MAP: dict[str, str] = {
    "经营业务现金净额": "operating_cash_flow",
    "经营活动产生的现金流量净额": "operating_cash_flow",
    "购买物业、厂房及设备": "capex",
    "购建固定资产": "capex",
    "已付股利": "dividends_paid",
    "已付股息": "dividends_paid",
    "已付股息(融资)": "dividends_paid",
    "回购股份": "buyback_amount",
    "偿还借贷": "debt_repaid",
    "偿还借款": "debt_repaid",
    "新增借贷": "debt_issued",
    "新增借款": "debt_issued",
    "收购附属公司": "acquisitions",
}


def fetch_hk_financials(symbol: str, years: int = 7) -> dict[str, Any]:
    """Fetch HK stock full three statements from AkShare (东财 source).

    Uses stock_financial_hk_report_em with long-format pivot.
    """
    import akshare as ak

    code = re.sub(r"[^\d]", "", symbol.split(".")[0]).zfill(5)
    result: dict[str, Any] = {
        "income_statement": [],
        "balance_sheet": [],
        "cash_flow": [],
        "source": "akshare_hk_em",
    }

    # --- Income Statement ---
    try:
        df = _akshare_call_with_retry(
            ak.stock_financial_hk_report_em, code, "利润表", "年度",
            label=f"HK IS {code}",
        )
        pivoted = _pivot_hk_long_format(df, years)

        for fiscal_year, items in sorted(pivoted.items()):
            row: dict[str, Any] = {"fiscal_year": fiscal_year, "fiscal_period": "FY"}
            for cn_name, field in _HK_IS_MAP.items():
                if cn_name in items and field not in row:
                    val = items[cn_name]
                    # cost fields are positive in source, negate for consistency
                    if field in ("cost_of_revenue", "interest_expense", "tax_provision", "sga_expense", "rd_expense"):
                        val = -abs(val) if val > 0 else val
                    row[field] = val
            row.setdefault("shares_basic", None)
            row.setdefault("shares_diluted", None)
            result["income_statement"].append(row)

        logger.info("AkShare HK IS: %d rows for %s", len(result["income_statement"]), code)
    except Exception:
        logger.warning("AkShare HK IS failed for %s", code, exc_info=True)

    # --- Balance Sheet ---
    try:
        df = _akshare_call_with_retry(
            ak.stock_financial_hk_report_em, code, "资产负债表", "年度",
            label=f"HK BS {code}",
        )
        pivoted = _pivot_hk_long_format(df, years)

        for fiscal_year, items in sorted(pivoted.items()):
            row = {"fiscal_year": fiscal_year}
            for cn_name, field in _HK_BS_MAP.items():
                if cn_name in items and field not in row:
                    row[field] = items[cn_name]
            result["balance_sheet"].append(row)

        logger.info("AkShare HK BS: %d rows for %s", len(result["balance_sheet"]), code)
    except Exception:
        logger.warning("AkShare HK BS failed for %s", code, exc_info=True)

    # --- Cash Flow ---
    try:
        df = _akshare_call_with_retry(
            ak.stock_financial_hk_report_em, code, "现金流量表", "年度",
            label=f"HK CF {code}",
        )
        pivoted = _pivot_hk_long_format(df, years)

        for fiscal_year, items in sorted(pivoted.items()):
            row: dict[str, Any] = {"fiscal_year": fiscal_year}
            for cn_name, field in _HK_CF_MAP.items():
                if cn_name in items and field not in row:
                    val = items[cn_name]
                    if field == "capex":
                        val = abs(val)
                    row[field] = val
            # Derive FCF
            ocf = row.get("operating_cash_flow")
            capex = row.get("capex")
            if ocf is not None and capex is not None:
                row["free_cash_flow"] = ocf - abs(capex)
            result["cash_flow"].append(row)

        logger.info("AkShare HK CF: %d rows for %s", len(result["cash_flow"]), code)
    except Exception:
        logger.warning("AkShare HK CF failed for %s", code, exc_info=True)

    return result


async def fetch_structured_financials(
    ticker: str, market: str, years: int = 7,
    akshare_cache: "AkShareCache | None" = None,
) -> dict[str, Any]:
    """Async wrapper: fetch structured financials from AkShare.

    Serialized via _AKSHARE_LOCK — py_mini_racer (V8) is not thread-safe.
    Returns empty dict if market not supported or API fails.

    If *akshare_cache* is provided, checks cache first and stores on miss.

    Circuit breaker: after N consecutive ticker failures, pauses for 5 minutes
    then retries (not permanent disable). Prevents wasting time on transient
    outages while allowing recovery.
    """
    global _CONSECUTIVE_FAILURES, _CIRCUIT_COOLDOWN_UNTIL

    # Check cache first
    if akshare_cache is not None:
        cached = akshare_cache.get(ticker, market)
        if cached is not None:
            logger.info("AkShare cache hit: %s (%s)", ticker, market)
            return cached

    # If in cooldown, check if it's time to retry
    if _CIRCUIT_COOLDOWN_UNTIL > 0:
        if time.time() < _CIRCUIT_COOLDOWN_UNTIL:
            return {}
        # Cooldown expired — reset and retry
        logger.info("AkShare circuit breaker cooldown expired, retrying...")
        _CONSECUTIVE_FAILURES = 0
        _CIRCUIT_COOLDOWN_UNTIL = 0.0

    # A-share: Sina source (no V8, thread-safe, no lock needed)
    # HK: eastmoney datacenter (still uses V8, needs lock)
    from investagent.executors import io_pool
    loop = asyncio.get_running_loop()
    if market == "A_SHARE":
        result = await loop.run_in_executor(
            io_pool(), fetch_a_share_financials, ticker, years,
        )
    elif market == "HK":
        async with _AKSHARE_LOCK:
            result = await loop.run_in_executor(
                io_pool(), fetch_hk_financials, ticker, years,
            )
    else:
        return {}

    # Circuit breaker: track consecutive failures
    has_data = any(result.get(k) for k in ("income_statement", "balance_sheet", "cash_flow"))
    if has_data:
        _CONSECUTIVE_FAILURES = 0
        # Store in cache on success
        if akshare_cache is not None:
            akshare_cache.put(ticker, market, result)
    else:
        _CONSECUTIVE_FAILURES += 1
        if _CONSECUTIVE_FAILURES >= _CIRCUIT_BREAKER_THRESHOLD:
            _CIRCUIT_COOLDOWN_UNTIL = time.time() + _CIRCUIT_COOLDOWN_SECS
            logger.warning(
                "AkShare circuit breaker: %d consecutive failures — "
                "pausing for %ds (not permanent)",
                _CONSECUTIVE_FAILURES, _CIRCUIT_COOLDOWN_SECS,
            )
    return result
