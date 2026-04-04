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
      cast_column   : cast column to float (coerce errors → NaN)
      coalesce      : replace NaN with default_value
      dedup         : keep first occurrence of column
      parse_currency: strip "$", "," then cast to float; "N/A" → NaN
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

        df = available.get(input_name)
        if df is None:
            # Cannot run this step — skip
            continue

        df = df.copy()

        # Apply filters
        for f in filters:
            df = apply_filter(df, f)

        # Apply patches
        for p in patches:
            df = apply_patch(df, p)

        # Aggregation steps: aggregate FIRST, then select output columns
        step_id = step.get("step_id", "")
        is_agg  = step_id in ("aggregate_summary", "aggregate_monthly",
                              "aggregate_daily", "rep_summary")

        if is_agg:
            df = _do_aggregation(df, step_id)
            # Column selection on aggregated output
            if select_cols:
                df = df[[c for c in select_cols if c in df.columns]]
        else:
            # Non-aggregation: select input columns first
            if select_cols:
                df = df[[c for c in select_cols if c in df.columns]]

        available[output_name] = df

    return available


def _do_aggregation(df: pd.DataFrame, step_id: str) -> pd.DataFrame:
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

    return df