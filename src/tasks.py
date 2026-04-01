"""
Task definitions for the Data Pipeline Incident Response environment.

Each task returns a dict that fully specifies:
  - Raw input tables (pandas DataFrames with injected faults)
  - DAG structure (steps, transformations)
  - Assertions to check
  - Historical schema (for drift detection)
  - Historical run metadata
  - Gold-standard fix info (used by grader, never shown to agent)
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd


# ============================================================
# TASK 1 — EASY
# ============================================================
# Fault: upstream raw_orders table suddenly has null user_ids
# (5 out of 100 rows).  A new nullable column "discount_code"
# was also added, which caused the upstream export to misfire
# and null-out user_id for those rows.
#
# Fix: add_data_filter on transform_orders step:
#         filter_condition = "user_id IS NOT NULL"
# ============================================================

def make_easy_task() -> Dict[str, Any]:
    np.random.seed(42)
    n = 100
    null_idx = {9, 24, 49, 74, 98}

    raw_orders = pd.DataFrame({
        "order_id":      list(range(1, n + 1)),
        "user_id":       [f"USR_{i:04d}" if i not in null_idx else None
                          for i in range(n)],
        "amount":        np.round(np.random.uniform(10, 500, n), 2).tolist(),
        "order_date":    ["2024-01-15"] * n,
        "discount_code": [None] * n,          # new column added by upstream
    })

    return {
        "task_id":    "easy",
        "difficulty": "easy",
        "description": (
            "Your nightly orders pipeline failed. The orders_clean table has a "
            "NOT NULL assertion on user_id that is now failing (5 of 100 rows). "
            "Investigate raw_orders, identify the root cause, and fix the pipeline."
        ),
        "raw_tables": {"raw_orders": raw_orders},
        "dag": [
            {
                "step_id":                  "transform_orders",
                "input_table":              "raw_orders",
                "output_table":             "orders_clean",
                "transformation_description": "SELECT order_id, user_id, amount, order_date FROM raw_orders",
                "select_columns":           ["order_id", "user_id", "amount", "order_date"],
                "applied_filters":          [],
                "applied_patches":          [],
            }
        ],
        "assertions": [
            {"id": "A1", "table": "orders_clean", "type": "not_null",  "column": "user_id"},
            {"id": "A2", "table": "orders_clean", "type": "unique",    "column": "order_id"},
            {"id": "A3", "table": "orders_clean", "type": "row_count", "min": 80, "max": 110},
        ],
        "schemas": {
            "raw_orders": {
                "current":    {"order_id": "int64",  "user_id": "object",
                               "amount":   "float64","order_date": "object",
                               "discount_code": "object"},
                "historical": {"order_id": "int64",  "user_id": "object",
                               "amount":   "float64","order_date": "object"},
            }
        },
        "historical_runs": [
            {"date": "2024-01-14", "status": "passed", "row_count": 98,  "duration_s": 12},
            {"date": "2024-01-13", "status": "passed", "row_count": 102, "duration_s": 11},
            {"date": "2024-01-12", "status": "passed", "row_count": 95,  "duration_s": 13},
        ],
        "accepted_assertions": [],
        "alerts_sent": [],
        # --- Gold standard (not shown to agent) ---
        "gold_root_cause": "null_user_id_upstream",
        "gold_fix_actions": [
            {"action_type": "add_data_filter",
             "params": {"step_id": "transform_orders",
                        "filter_condition": "user_id IS NOT NULL"}},
        ],
        "must_alert_upstream": False,
    }


# ============================================================
# TASK 2 — MEDIUM
# ============================================================
# Fault: vendor sent duplicate order_item_id values (20 dupes
# out of 200 rows).  This causes:
#   - B1 (unique order_item_id) to fail on order_items_clean
#   - B3 (row_count 95–115) on order_summary to be too high
#     because SUM is inflated by duplicate rows
#
# Fix:
#   1. read_data_sample / check_schema to diagnose
#   2. patch_transformation on transform_items:
#        patch_type=dedup, column=order_item_id
#   3. run_pipeline  →  both assertions should now pass
# ============================================================

def make_medium_task() -> Dict[str, Any]:
    np.random.seed(7)
    n_base = 200
    n_dup  = 20   # 20 duplicate rows injected

    base = pd.DataFrame({
        "order_item_id": list(range(1, n_base + 1)),
        "order_id":      np.random.randint(1, 51, n_base).tolist(),
        "product_id":    np.random.randint(100, 200, n_base).tolist(),
        "quantity":      np.random.randint(1, 5, n_base).tolist(),
        "unit_price":    np.round(np.random.uniform(5, 200, n_base), 2).tolist(),
        "created_at":    ["2024-01-15"] * n_base,
    })

    # Duplicate 20 random rows (simulate vendor re-sending them)
    dup_rows = base.sample(n_dup, random_state=99).copy()
    raw_order_items = pd.concat([base, dup_rows], ignore_index=True).sample(
        frac=1, random_state=13
    ).reset_index(drop=True)

    # Reference orders table (clean)
    raw_orders = pd.DataFrame({
        "order_id":   list(range(1, 51)),
        "customer_id": [f"CUST_{i:03d}" for i in range(1, 51)],
        "order_date": ["2024-01-15"] * 50,
    })

    return {
        "task_id":    "medium",
        "difficulty": "medium",
        "description": (
            "Two assertions are failing in your order-items pipeline. "
            "The order_items_clean table has a uniqueness failure on order_item_id, "
            "and the order_summary row count is out of range. "
            "Vendor data arrived overnight. Investigate and fix."
        ),
        "raw_tables": {
            "raw_order_items": raw_order_items,
            "raw_orders":      raw_orders,
        },
        "dag": [
            {
                "step_id":                  "transform_items",
                "input_table":              "raw_order_items",
                "output_table":             "order_items_clean",
                "transformation_description": (
                    "SELECT order_item_id, order_id, product_id, quantity, "
                    "unit_price FROM raw_order_items"
                ),
                "select_columns": ["order_item_id", "order_id", "product_id",
                                   "quantity", "unit_price"],
                "applied_filters": [],
                "applied_patches": [],
            },
            {
                "step_id":                  "aggregate_summary",
                "input_table":              "order_items_clean",
                "output_table":             "order_summary",
                "transformation_description": (
                    "SELECT order_id, COUNT(*) as item_count, "
                    "SUM(quantity*unit_price) as total FROM order_items_clean "
                    "GROUP BY order_id"
                ),
                "select_columns": ["order_id", "item_count", "total"],
                "applied_filters": [],
                "applied_patches": [],
            },
        ],
        "assertions": [
            {"id": "B1", "table": "order_items_clean", "type": "unique",
             "column": "order_item_id"},
            {"id": "B2", "table": "order_items_clean", "type": "not_null",
             "column": "order_id"},
            {"id": "B3", "table": "order_items_clean", "type": "row_count",
             "min": 195, "max": 205},   # 220 rows initially (200 + 20 dupes) → FAILS
            {"id": "B4", "table": "order_items_clean", "type": "value_range",
             "column": "unit_price", "min": 0, "max": 10000},
        ],
        "schemas": {
            "raw_order_items": {
                "current":    {"order_item_id": "int64", "order_id": "int64",
                               "product_id": "int64",   "quantity": "int64",
                               "unit_price": "float64", "created_at": "object"},
                "historical": {"order_item_id": "int64", "order_id": "int64",
                               "product_id": "int64",   "quantity": "int64",
                               "unit_price": "float64", "created_at": "object"},
            }
        },
        "historical_runs": [
            {"date": "2024-01-14", "status": "passed", "row_count": 198, "duration_s": 22},
            {"date": "2024-01-13", "status": "passed", "row_count": 203, "duration_s": 21},
            {"date": "2024-01-12", "status": "passed", "row_count": 195, "duration_s": 23},
        ],
        "accepted_assertions": [],
        "alerts_sent": [],
        # --- Gold standard ---
        "gold_root_cause": "duplicate_order_item_ids_from_vendor",
        "gold_fix_actions": [
            {"action_type": "patch_transformation",
             "params": {"step_id": "transform_items",
                        "patch_type": "dedup",
                        "column": "order_item_id"}},
        ],
        "must_alert_upstream": False,
    }


# ============================================================
# TASK 3 — HARD
# ============================================================
# Fault: Salesforce changed the "revenue" field from a
# formatted string ("$1,234.56") to a plain float (1234.56)
# on 2024-01-08 (mid-month).
#
# Before 2024-01-08: revenue = "$1,234.56"   (string)
# After  2024-01-08: revenue = 1234.56        (float stored as string)
# ALSO: 12 rows where revenue = "N/A" (genuinely corrupted,
#       cannot be fixed locally — must alert upstream).
#
# 6 assertions fail across 4 tables:
#   C1 type_check  on crm_clean.revenue        (mixed types)
#   C2 value_range on crm_clean.revenue        (strings fail numeric range)
#   C3 not_null    on revenue_monthly.amount   (SUM fails → nulls propagate)
#   C4 row_count   on revenue_monthly          (N/A rows may be dropped)
#   C5 not_null    on crm_clean.account_id     (red herring — always passes)
#   C6 value_range on revenue_daily.daily_total (inflated by string concat)
#
# Fix:
#   1. Read data samples — see mixed formats
#   2. Compare schema — revenue was "object" historically too (no schema diff)
#   3. patch_transformation on transform_crm:
#        patch_type=parse_currency, column=revenue
#      (strips $, commas, casts to float; sets N/A → null)
#   4. alert_upstream_team for the N/A rows
#   5. run_pipeline  →  C1, C2, C3, C4, C6 pass; C5 was already passing
# ============================================================

def make_hard_task() -> Dict[str, Any]:
    np.random.seed(2024)

    n = 300
    dates = []
    for i in range(n):
        day = (i % 31) + 1
        dates.append(f"2024-01-{day:02d}")

    revenues = []
    for i, d in enumerate(dates):
        day = int(d.split("-")[2])
        amount = round(np.random.uniform(500, 50000), 2)
        if i < 12:
            # Genuinely corrupted rows
            revenues.append("N/A")
        elif day < 8:
            # Old format: formatted dollar string
            revenues.append(f"${amount:,.2f}")
        else:
            # New format: plain number as string
            revenues.append(str(amount))

    raw_crm = pd.DataFrame({
        "account_id":   [f"ACC_{i:05d}" for i in range(n)],
        "account_name": [f"Company {i}" for i in range(n)],
        "revenue":      revenues,
        "region":       np.random.choice(["NA", "EU", "APAC"], n).tolist(),
        "close_date":   dates,
        "rep_id":       np.random.randint(1, 20, n).tolist(),
    })

    return {
        "task_id":    "hard",
        "difficulty": "hard",
        "description": (
            "6 assertions are failing across 4 tables in your CRM revenue pipeline. "
            "The Salesforce export changed format mid-month. Some records are genuinely "
            "corrupted (N/A values) and some are fixable locally. "
            "You must distinguish between the two: fix what you can, "
            "alert upstream for what you cannot."
        ),
        "raw_tables": {"raw_crm": raw_crm},
        "dag": [
            {
                "step_id":       "transform_crm",
                "input_table":   "raw_crm",
                "output_table":  "crm_clean",
                "transformation_description": (
                    "SELECT account_id, account_name, CAST(revenue AS FLOAT) as revenue, "
                    "region, close_date FROM raw_crm"
                ),
                "select_columns": ["account_id", "account_name", "revenue",
                                   "region", "close_date", "rep_id"],
                "applied_filters": [],
                "applied_patches": [],
            },
            {
                "step_id":       "aggregate_monthly",
                "input_table":   "crm_clean",
                "output_table":  "revenue_monthly",
                "transformation_description": (
                    "SELECT LEFT(close_date,7) as month, SUM(revenue) as amount, "
                    "COUNT(*) as deal_count FROM crm_clean GROUP BY month"
                ),
                "select_columns": ["month", "amount", "deal_count"],
                "applied_filters": [],
                "applied_patches": [],
            },
            {
                "step_id":       "aggregate_daily",
                "input_table":   "crm_clean",
                "output_table":  "revenue_daily",
                "transformation_description": (
                    "SELECT close_date, SUM(revenue) as daily_total "
                    "FROM crm_clean GROUP BY close_date"
                ),
                "select_columns": ["close_date", "daily_total"],
                "applied_filters": [],
                "applied_patches": [],
            },
            {
                "step_id":       "rep_summary",
                "input_table":   "crm_clean",
                "output_table":  "rep_performance",
                "transformation_description": (
                    "SELECT rep_id, SUM(revenue) as total_revenue, "
                    "COUNT(*) as deal_count FROM crm_clean GROUP BY rep_id"
                ),
                "select_columns": ["rep_id", "total_revenue", "deal_count"],
                "applied_filters": [],
                "applied_patches": [],
            },
        ],
        "assertions": [
            {"id": "C1", "table": "crm_clean",       "type": "type_check",
             "column": "revenue", "expected_type": "numeric"},
            # C2: NaN (from N/A rows after parse_currency) counts as out-of-range
            {"id": "C2", "table": "crm_clean",       "type": "value_range",
             "column": "revenue", "min": 0, "max": 1_000_000},
            # C3: SUM is 0 initially (all strings → NaN → sum=0)
            {"id": "C3", "table": "revenue_monthly", "type": "value_range",
             "column": "amount",  "min": 100_000, "max": 500_000_000},
            # C4: COUNT of non-null revenue is 0 initially (all parse to NaN)
            {"id": "C4", "table": "revenue_monthly", "type": "value_range",
             "column": "deal_count", "min": 200, "max": 400},
            # C5: RED HERRING — account_id is never null, always passes
            {"id": "C5", "table": "crm_clean",       "type": "not_null",
             "column": "account_id"},
            # C6: daily totals are 0 initially (string revenues)
            {"id": "C6", "table": "revenue_daily",   "type": "value_range",
             "column": "daily_total", "min": 1_000, "max": 50_000_000},
        ],
        "schemas": {
            "raw_crm": {
                "current":    {"account_id": "object", "account_name": "object",
                               "revenue": "object",    "region": "object",
                               "close_date": "object", "rep_id": "int64"},
                "historical": {"account_id": "object", "account_name": "object",
                               "revenue": "object",    "region": "object",
                               "close_date": "object", "rep_id": "int64"},
            }
        },
        "historical_runs": [
            {"date": "2024-01-07", "status": "passed", "row_count": 298, "duration_s": 45},
            {"date": "2024-01-06", "status": "passed", "row_count": 301, "duration_s": 43},
            {"date": "2024-01-05", "status": "passed", "row_count": 295, "duration_s": 46},
        ],
        "accepted_assertions": [],
        "alerts_sent": [],
        # --- Gold standard ---
        "gold_root_cause": "revenue_format_change_and_corrupt_na_rows",
        "gold_fix_actions": [
            {"action_type": "patch_transformation",
             "params": {"step_id": "transform_crm",
                        "patch_type": "parse_currency",
                        "column": "revenue"}},
            {"action_type": "alert_upstream_team",
             "params": {"team": "salesforce_ops",
                        "issue_description": "12 rows with revenue=N/A cannot be parsed"}},
        ],
        "must_alert_upstream": True,
        "upstream_team_to_alert": "salesforce_ops",
    }


# ============================================================
# Registry
# ============================================================

TASKS: Dict[str, Any] = {
    "easy":   make_easy_task,
    "medium": make_medium_task,
    "hard":   make_hard_task,
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Choose from {list(TASKS)}")
    return TASKS[task_id]()