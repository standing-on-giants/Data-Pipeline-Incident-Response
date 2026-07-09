"""
Pipeline runner.
Given raw tables + current DAG state (filters / patches applied so far),
produces output tables by executing each step in order.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd


# ------------------------------------------------------------------ #
# Filter application
# ------------------------------------------------------------------ #

def apply_filter(df: pd.DataFrame, filter_condition: str) -> pd.DataFrame:
    """
    Apply a single WHERE-style filter to a DataFrame.
    Supported conditions (parsed by simple keyword matching):
      - "column IS NOT NULL"
      - "column IS NULL"
      - "column > value"
      - "column < value"
      - Custom: "ROW_NUMBER_DEDUP:column" (kept internally, real dedup done via patch)
    """
    cond = filter_condition.strip()

    if "IS NOT NULL" in cond:
        col = cond.replace("IS NOT NULL", "").strip()
        if col in df.columns:
            return df[df[col].notna()].reset_index(drop=True)

    elif "IS NULL" in cond:
        col = cond.replace("IS NULL", "").strip()
        if col in df.columns:
            return df[df[col].isna()].reset_index(drop=True)

    elif ">=" in cond:
        parts = cond.split(">=")
        col, val = parts[0].strip(), parts[1].strip()
        if col in df.columns:
            try:
                return df[pd.to_numeric(df[col], errors="coerce") >= float(val)].reset_index(drop=True)
            except Exception:
                pass

    elif "<=" in cond:
        parts = cond.split("<=")
        col, val = parts[0].strip(), parts[1].strip()
        if col in df.columns:
            try:
                return df[pd.to_numeric(df[col], errors="coerce") <= float(val)].reset_index(drop=True)
            except Exception:
                pass

    # Unknown filter — return unchanged (log-worthy in production) 
    return df


# ------------------------------------------------------------------ #
# Patch application
# ------------------------------------------------------------------ #

def apply_patch(df: pd.DataFrame, patch: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply a column-level transformation patch.

    Supported patch_types:
      cast_column   : cast column to float (coerce errors -> NaN)
      coalesce      : replace NaN with default_value
      dedup         : keep first occurrence of column
      parse_currency: strip "$", "," then cast to float; "N/A" -> NaN
      strip_prefix  : strip a string prefix (e.g. "CMP_") from values
    """
    p_type  = patch.get("patch_type", "")
    col     = patch.get("column")
    result  = df.copy()

    if col not in result.columns:
        return result   # silently ignore unknown column

    if p_type == "cast_column":
        result[col] = pd.to_numeric(result[col], errors="coerce")

    elif p_type == "coalesce":
        default = patch.get("default_value") or 0
        result[col] = result[col].fillna(default)

    elif p_type == "dedup":
        result = result.drop_duplicates(subset=[col], keep="first").reset_index(drop=True)

    elif p_type == "parse_currency":
        def _parse(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return float("nan")
            s = str(v).strip()
            if s in ("N/A", "NA", "n/a", "", "null"):
                return float("nan")
            # Strip currency symbols and commas
            s = s.replace("$", "").replace(",", "").strip()
            try:
                return float(s)
            except ValueError:
                return float("nan")

        result[col] = result[col].apply(_parse)

    elif p_type == "strip_prefix":
        prefix = patch.get("prefix", "CMP_")
        def _strip(v):
            if v is None:
                return v
            try:
                if pd.isna(v):
                    return v
            except (TypeError, ValueError):
                pass
            s = str(v).strip()
            if s.startswith(prefix):
                return s[len(prefix):]
            return s
        result[col] = result[col].apply(_strip)

    return result


# ------------------------------------------------------------------ #
# Computed columns (e.g. CTR = clicks / impressions)
# ------------------------------------------------------------------ #

def _compute_columns(df: pd.DataFrame, computed: List[Dict[str, str]]) -> pd.DataFrame:
    """Evaluate computed column expressions after patches are applied."""
    result = df.copy()
    for spec in computed:
        name    = spec["name"]
        formula = spec["formula"]
        # Supported: "col_a / col_b"
        if "/" in formula:
            parts = [p.strip() for p in formula.split("/")]
            if len(parts) == 2 and parts[0] in result.columns and parts[1] in result.columns:
                num = pd.to_numeric(result[parts[0]], errors="coerce")
                den = pd.to_numeric(result[parts[1]], errors="coerce")
                result[name] = num / den  # produces Inf / NaN on zero
    return result


# ------------------------------------------------------------------ #
# Full pipeline execution
# ------------------------------------------------------------------ #

def execute_pipeline(
    raw_tables: Dict[str, pd.DataFrame],
    dag: List[Dict[str, Any]],
) -> Dict[str, pd.DataFrame]:
    """
    Execute each DAG step in order, returning all output tables.
    raw_tables are never modified.
    """
    available: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in raw_tables.items()}

    for step in dag:
        input_name  = step["input_table"]
        output_name = step["output_table"]
        select_cols = step.get("select_columns")
        filters     = step.get("applied_filters", [])
        patches     = step.get("applied_patches", [])
        computed    = step.get("computed_columns", [])

        df = available.get(input_name)
        if df is None:
            # Cannot run this step -- skip
            continue

        df = df.copy()

        # Join support: merge with another table before processing
        join_table = step.get("join_table")
        join_key   = step.get("join_key")
        if join_table and join_key:
            df2 = available.get(join_table)
            if df2 is not None:
                df = pd.merge(df, df2, on=join_key, how="inner")

        # Apply patches first (e.g. parse_currency converts "N/A" -> NaN)
        for p in patches:
            df = apply_patch(df, p)

        # Apply filters after patches (so IS NOT NULL can catch NaN from parse_currency)
        for f in filters:
            df = apply_filter(df, f)

        # Apply computed columns (e.g. CTR = clicks / impressions)
        if computed:
            df = _compute_columns(df, computed)

        # Aggregation steps: aggregate FIRST, then select output columns
        step_id = step.get("step_id", "")
        is_agg  = step_id in ("aggregate_summary", "aggregate_monthly",
                              "aggregate_daily", "rep_summary",
                              "calculate_roas")

        if is_agg:
            df = _do_aggregation(df, step_id, available)
            # Column selection on aggregated output
            if select_cols:
                df = df[[c for c in select_cols if c in df.columns]]
        else:
            # Non-aggregation: select input columns first
            if select_cols:
                df = df[[c for c in select_cols if c in df.columns]]

        available[output_name] = df

    return available


def _do_aggregation(
    df: pd.DataFrame, step_id: str, available: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Inline aggregation logic keyed by step_id."""

    if step_id == "aggregate_summary":
        # order_summary: group by order_id, count items, sum revenue
        if "order_id" in df.columns and "unit_price" in df.columns:
            df["line_total"] = pd.to_numeric(df["quantity"], errors="coerce") * \
                               pd.to_numeric(df["unit_price"], errors="coerce")
            agg = df.groupby("order_id", as_index=False).agg(
                item_count=("order_id", "count"),
                total=("line_total", "sum"),
            )
            return agg

    elif step_id == "aggregate_monthly":
        # revenue_monthly: group by month(close_date), sum revenue
        if "close_date" in df.columns and "revenue" in df.columns:
            df = df.copy()
            df["month"] = df["close_date"].str[:7]
            df["revenue_num"] = pd.to_numeric(df["revenue"], errors="coerce")
            agg = df.groupby("month", as_index=False).agg(
                amount=("revenue_num", "sum"),
                deal_count=("revenue_num", "count"),
            )
            return agg

    elif step_id == "aggregate_daily":
        # revenue_daily: group by close_date, sum revenue
        if "close_date" in df.columns and "revenue" in df.columns:
            df = df.copy()
            df["revenue_num"] = pd.to_numeric(df["revenue"], errors="coerce")
            agg = df.groupby("close_date", as_index=False).agg(
                daily_total=("revenue_num", "sum"),
            )
            return agg

    elif step_id == "rep_summary":
        # rep_performance: group by rep_id, sum revenue
        if "rep_id" in df.columns and "revenue" in df.columns:
            df = df.copy()
            df["revenue_num"] = pd.to_numeric(df["revenue"], errors="coerce")
            agg = df.groupby("rep_id", as_index=False).agg(
                total_revenue=("revenue_num", "sum"),
                deal_count=("revenue_num", "count"),
            )
            return agg

    elif step_id == "calculate_roas":
        # ROAS: join clean_insights with clean_conversions on campaign_id
        df_conversions = available.get("clean_conversions")
        empty_roas = pd.DataFrame(columns=[
            "campaign_id", "total_spend", "total_revenue",
            "purchase_count", "roas",
        ])

        if df_conversions is None or "campaign_id" not in df.columns:
            return empty_roas
        if "campaign_id" not in df_conversions.columns:
            return empty_roas

        # SAFE MERGE: cast both sides to str so pd.merge never crashes
        # on int64 vs object mismatch.  When the CMP_ prefix hasn't been
        # stripped, the keys simply won't match -> 0-row result (intended bug).
        df_ins = df.copy()
        df_con = df_conversions.copy()
        df_ins["_join_key"] = df_ins["campaign_id"].astype(str)
        df_con["_join_key"] = df_con["campaign_id"].astype(str)

        joined = pd.merge(df_ins, df_con, on="_join_key", how="inner",
                          suffixes=("", "_conv"))

        if joined.empty:
            return empty_roas

        joined["spend_num"] = pd.to_numeric(joined["spend"], errors="coerce")
        joined["pv_num"]    = pd.to_numeric(joined["purchase_value"], errors="coerce")

        agg = joined.groupby("_join_key", as_index=False).agg(
            total_spend=("spend_num", "sum"),
            total_revenue=("pv_num", "sum"),
            purchase_count=("_join_key", "count"),
        )
        agg = agg.rename(columns={"_join_key": "campaign_id"})

        # ROAS = revenue / spend  (guard div-by-zero)
        agg["roas"] = agg.apply(
            lambda r: r["total_revenue"] / r["total_spend"]
            if r["total_spend"] and r["total_spend"] > 0 else 0.0,
            axis=1,
        )
        agg["roas"] = agg["roas"].replace(
            [float("inf"), -float("inf")], float("nan")
        ).fillna(0)

        return agg

    return df
