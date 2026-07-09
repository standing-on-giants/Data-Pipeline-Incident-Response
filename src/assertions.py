"""
Deterministic assertion checker.
All assertions return a float score in [0.0, 1.0] and a boolean pass/fail.
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd
from src.models import AssertionResult


def check_assertion(
    tables: Dict[str, pd.DataFrame],
    assertion: Dict[str, Any],
) -> AssertionResult:
    """
    Run a single assertion against the current set of tables.
    Returns an AssertionResult with pass/fail and failing row count.
    """
    a_id   = assertion["id"]
    a_type = assertion["type"]
    table  = assertion["table"]
    col    = assertion.get("column")

    df = tables.get(table)
    if df is None:
        return AssertionResult(
            assertion_id=a_id, table=table, assertion_type=a_type,
            column=col, expected="table exists", actual="table not found",
            passed=False, failing_row_count=0,
        )

    # ------------------------------------------------------------------
    # NOT NULL
    # ------------------------------------------------------------------
    if a_type == "not_null":
        null_count = int(df[col].isna().sum())
        passed     = null_count == 0
        return AssertionResult(
            assertion_id=a_id, table=table, assertion_type=a_type,
            column=col,
            expected="0 null rows",
            actual=f"{null_count} null rows",
            passed=passed,
            failing_row_count=null_count,
        )

    # ------------------------------------------------------------------
    # UNIQUE
    # ------------------------------------------------------------------
    if a_type == "unique":
        dup_count = int(df[col].duplicated().sum())
        passed    = dup_count == 0
        return AssertionResult(
            assertion_id=a_id, table=table, assertion_type=a_type,
            column=col,
            expected="0 duplicate rows",
            actual=f"{dup_count} duplicate rows",
            passed=passed,
            failing_row_count=dup_count,
        )

    # ------------------------------------------------------------------
    # ROW COUNT
    # ------------------------------------------------------------------
    if a_type == "row_count":
        actual_rows = len(df)
        lo          = assertion.get("min", 0)
        hi          = assertion.get("max", 10_000_000)
        passed      = lo <= actual_rows <= hi
        return AssertionResult(
            assertion_id=a_id, table=table, assertion_type=a_type,
            column=None,
            expected=f"between {lo} and {hi} rows",
            actual=f"{actual_rows} rows",
            passed=passed,
            failing_row_count=0 if passed else abs(actual_rows - (lo if actual_rows < lo else hi)),
        )

    # ------------------------------------------------------------------
    # VALUE RANGE  (numeric column must be within [min, max])
    # ------------------------------------------------------------------
    if a_type == "value_range":
        lo = assertion.get("min")
        hi = assertion.get("max")
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")
            out_of_range = 0
            if lo is not None:
                out_of_range += int((numeric < lo).sum())
            if hi is not None:
                out_of_range += int((numeric > hi).sum())
            # also count coercion failures as failures
            out_of_range += int(numeric.isna().sum())
            passed = out_of_range == 0
        except Exception:
            out_of_range = len(df)
            passed = False
        return AssertionResult(
            assertion_id=a_id, table=table, assertion_type=a_type,
            column=col,
            expected=f"all values in [{lo}, {hi}]",
            actual=f"{out_of_range} values out of range or non-numeric",
            passed=passed,
            failing_row_count=out_of_range,
        )

    # ------------------------------------------------------------------
    # TYPE CHECK  (non-null values must be parseable as numeric)
    # ------------------------------------------------------------------
    if a_type == "type_check":
        expected_type = assertion.get("expected_type", "numeric")
        if expected_type == "numeric":
            # Only flag values that are non-null AND cannot be parsed as a number
            non_null = df[col].dropna()
            bad = int(pd.to_numeric(non_null, errors="coerce").isna().sum())
        else: 
            bad = 0
        passed = bad == 0
        return AssertionResult(
            assertion_id=a_id, table=table, assertion_type=a_type,
            column=col,
            expected=f"all non-null values parseable as {expected_type}",
            actual=f"{bad} non-parseable non-null values",
            passed=passed,
            failing_row_count=bad,
        )

    # ------------------------------------------------------------------
    # Unknown assertion type — always fail
    # ------------------------------------------------------------------
    return AssertionResult(
        assertion_id=a_id, table=table, assertion_type=a_type,
        column=col,
        expected="known assertion type",
        actual=f"unknown type: {a_type}",
        passed=False, failing_row_count=0,
    )


def score_assertions(results: list[AssertionResult]) -> float:
    """Return a scalar score in [0.0, 1.0] from a list of assertion results."""
    if not results:
        return 1.0
    return sum(1 for r in results if r.passed) / len(results)
