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
#   - B3 (row_count 95-115) on order_summary to be too high
#     because SUM is inflated by duplicate rows
#
# Fix:
#   1. read_data_sample / check_schema to diagnose
#   2. patch_transformation on transform_items:
#        patch_type=dedup, column=order_item_id
#   3. run_pipeline  ->  both assertions should now pass
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
             "min": 195, "max": 205},   # 220 rows initially (200 + 20 dupes) -> FAILS
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
# TASK 3 — HARD: Meta Ads & Conversions Pipeline
# ============================================================
# Multi-stage pipeline ingesting Graph API (Ads Insights) and
# Conversions API (Purchase Events), joining them to compute
# ROAS (Return on Ad Spend).
#
# SEQUENTIAL FAILURES (agent discovers them one by one):
#
# State 0 (Initial):
#   `spend` arrives as "$X,XXX.XX" strings (Graph API v19.0
#   schema drift).  type_check on spend fails.
#   Fix: parse_currency on spend, then coalesce on spend.
#
# State 1 (Unlocked after spend is fixed):
#   `impressions` has "N/A" values from Graph API outage.
#   After being parsed and coalesced to 0, CTR = clicks/0
#   produces Inf/NaN.
#   Fix: parse_currency on impressions, then
#        coalesce(impressions, 1) — using 1 to avoid div/0.
#
# State 2 (Unlocked after CTR is fixed):
#   CAPI retried failed payloads overnight -> ~37 duplicate
#   events (15%).  Uniqueness assertion on event_id fails.
#   Fix: dedup on event_id.
#
# State 3 (Unlocked after dedup):
#   campaign_id in raw_conversions is "CMP_123" (string with
#   prefix), but raw_ads_insights uses plain int.  Inner join
#   silently drops 90%+ of rows -> roas_summary row_count
#   fails.
#   Fix: strip_prefix + cast_column on conversions campaign_id.
#
# Also: alert meta_ads_api_team about the Graph API outage
# that produced N/A impressions rows.
# ============================================================

def make_hard_task() -> Dict[str, Any]:
    np.random.seed(2024)

    n_insights = 200
    n_conv_base = 250
    n_conv_dupes = 37       # ~15% duplicate events
    n_campaigns = 50        # campaign IDs 1..50

    # ---- raw_ads_insights (200 rows) --------------------------------
    campaign_ids_ins = np.random.randint(1, n_campaigns + 1, n_insights).tolist()
    ad_names = [f"Ad_Group_{i}" for i in range(n_insights)]
    clicks   = np.random.randint(10, 5000, n_insights).tolist()

    # Impressions: mostly int, but ~8 rows are "N/A" (Graph API outage)
    na_imp_idx = set(np.random.choice(n_insights, 8, replace=False).tolist())
    impressions_raw = []
    for i in range(n_insights):
        if i in na_imp_idx:
            impressions_raw.append("N/A")
        else:
            impressions_raw.append(int(np.random.randint(1000, 500_000)))

    # Spend: mixed format — old "$X,XXX.XX", new plain float string, ~10 N/A
    na_spend_idx = set(np.random.choice(n_insights, 10, replace=False).tolist())
    spend_raw = []
    for i in range(n_insights):
        amount = round(np.random.uniform(50, 10_000), 2)
        if i in na_spend_idx:
            spend_raw.append("N/A")
        elif i % 3 == 0:
            spend_raw.append(f"${amount:,.2f}")
        else:
            spend_raw.append(str(amount))

    raw_ads_insights = pd.DataFrame({
        "campaign_id":  campaign_ids_ins,
        "ad_name":      ad_names,
        "spend":        spend_raw,
        "impressions":  impressions_raw,
        "clicks":       clicks,
    })

    # ---- raw_conversions (250 base + 37 dupes) ----------------------
    event_ids_base = [f"evt_{i:06d}" for i in range(n_conv_base)]
    campaign_ids_conv = [
        f"CMP_{np.random.randint(1, n_campaigns + 1)}"
        for _ in range(n_conv_base)
    ]
    event_times = [
        f"2024-01-{(i % 31) + 1:02d}T"
        f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}:00"
        for i in range(n_conv_base)
    ]
    purchase_values = np.round(
        np.random.uniform(10, 2000, n_conv_base), 2
    ).tolist()

    base_conv = pd.DataFrame({
        "event_id":       event_ids_base,
        "campaign_id":    campaign_ids_conv,
        "event_time":     event_times,
        "purchase_value": purchase_values,
    })

    # Duplicate ~37 random rows (CAPI retries)
    dup_rows = base_conv.sample(n_conv_dupes, random_state=99).copy()
    raw_conversions = pd.concat(
        [base_conv, dup_rows], ignore_index=True
    ).sample(frac=1, random_state=13).reset_index(drop=True)

    return {
        "task_id":    "hard",
        "difficulty": "hard",
        "description": (
            "Your Meta Ads ROAS pipeline failed this morning. "
            "The pipeline ingests Graph API Ads Insights and Conversions API "
            "purchase events, joins them by campaign_id, and computes ROAS. "
            "Multiple assertions are failing across clean_insights, "
            "clean_conversions, and roas_summary. The Graph API recently "
            "upgraded to v19.0, and the Conversions API retried failed payloads "
            "overnight. Investigate the data, fix what you can, and alert the "
            "upstream team about genuinely corrupted data."
        ),
        "raw_tables": {
            "raw_ads_insights": raw_ads_insights,
            "raw_conversions":  raw_conversions,
        },
        "dag": [
            {
                "step_id":       "transform_insights",
                "input_table":   "raw_ads_insights",
                "output_table":  "clean_insights",
                "transformation_description": (
                    "SELECT campaign_id, ad_name, CAST(spend AS FLOAT) as spend, "
                    "impressions, clicks, (clicks / impressions) AS ctr "
                    "FROM raw_ads_insights"
                ),
                "select_columns": ["campaign_id", "ad_name", "spend",
                                   "impressions", "clicks", "ctr"],
                "computed_columns": [
                    {"name": "ctr", "formula": "clicks / impressions"},
                ],
                "applied_filters": [],
                "applied_patches": [],
            },
            {
                "step_id":       "transform_conversions",
                "input_table":   "raw_conversions",
                "output_table":  "clean_conversions",
                "transformation_description": (
                    "SELECT event_id, campaign_id, event_time, purchase_value "
                    "FROM raw_conversions"
                ),
                "select_columns": ["event_id", "campaign_id", "event_time",
                                   "purchase_value"],
                "applied_filters": [],
                "applied_patches": [],
            },
            {
                "step_id":       "calculate_roas",
                "input_table":   "clean_insights",
                "output_table":  "roas_summary",
                "transformation_description": (
                    "SELECT campaign_id, SUM(spend) as total_spend, "
                    "SUM(purchase_value) as total_revenue, "
                    "COUNT(*) as purchase_count, "
                    "total_revenue / total_spend as roas "
                    "FROM clean_insights JOIN clean_conversions "
                    "USING(campaign_id) GROUP BY campaign_id"
                ),
                "select_columns": ["campaign_id", "total_spend",
                                   "total_revenue", "purchase_count", "roas"],
                "applied_filters": [],
                "applied_patches": [],
            },
        ],
        "assertions": [
            # -- clean_insights assertions --------------------------------
            # H1: spend must be numeric (fails initially — strings)
            {"id": "H1", "table": "clean_insights", "type": "type_check",
             "column": "spend", "expected_type": "numeric"},
            # H2: spend in valid range (NaN from N/A counts as out-of-range)
            {"id": "H2", "table": "clean_insights", "type": "value_range",
             "column": "spend", "min": 0, "max": 1_000_000},
            # H3: CTR in [0, 1] — div-by-zero produces NaN/Inf
            {"id": "H3", "table": "clean_insights", "type": "value_range",
             "column": "ctr", "min": 0, "max": 1},
            # H8: ad_name not null — RED HERRING, always passes
            {"id": "H8", "table": "clean_insights", "type": "not_null",
             "column": "ad_name"},

            # -- clean_conversions assertions ----------------------------
            # H4: event_id must be unique (fails — CAPI retries)
            {"id": "H4", "table": "clean_conversions", "type": "unique",
             "column": "event_id"},
            # H5: row count 230-260 (287 with dupes -> fails)
            {"id": "H5", "table": "clean_conversions", "type": "row_count",
             "min": 230, "max": 260},

            # -- roas_summary assertions ---------------------------------
            # H6: should have 15-50 campaigns; join failure -> near 0
            {"id": "H6", "table": "roas_summary", "type": "row_count",
             "min": 15, "max": 55},
            # H7: ROAS in reasonable range
            {"id": "H7", "table": "roas_summary", "type": "value_range",
             "column": "roas", "min": 0, "max": 100},
        ],
        "schemas": {
            "raw_ads_insights": {
                "current":    {"campaign_id": "int64",  "ad_name": "object",
                               "spend": "object",       "impressions": "object",
                               "clicks": "int64"},
                "historical": {"campaign_id": "int64",  "ad_name": "object",
                               "spend": "float64",      "impressions": "int64",
                               "clicks": "int64"},
            },
            "raw_conversions": {
                "current":    {"event_id": "object",    "campaign_id": "object",
                               "event_time": "object",  "purchase_value": "float64"},
                "historical": {"event_id": "object",    "campaign_id": "int64",
                               "event_time": "object",  "purchase_value": "float64"},
            },
        },
        "historical_runs": [
            {"date": "2024-01-14", "status": "passed", "row_count": 198, "duration_s": 35},
            {"date": "2024-01-13", "status": "passed", "row_count": 201, "duration_s": 33},
            {"date": "2024-01-12", "status": "passed", "row_count": 195, "duration_s": 36},
        ],
        "accepted_assertions": [],
        "alerts_sent": [],
        # --- Gold standard (not shown to agent) ---
        "gold_root_cause": "graph_api_v19_schema_drift_capi_retries_join_key_mismatch",
        "gold_fix_actions": [
            {"action_type": "patch_transformation",
             "params": {"step_id": "transform_insights",
                        "patch_type": "parse_currency",
                        "column": "spend"}},
            {"action_type": "patch_transformation",
             "params": {"step_id": "transform_insights",
                        "patch_type": "coalesce",
                        "column": "spend"}},
            {"action_type": "patch_transformation",
             "params": {"step_id": "transform_insights",
                        "patch_type": "parse_currency",
                        "column": "impressions"}},
            {"action_type": "add_data_filter",
             "params": {"step_id": "transform_insights",
                        "filter_condition": "impressions IS NOT NULL"}},
            {"action_type": "patch_transformation",
             "params": {"step_id": "transform_conversions",
                        "patch_type": "dedup",
                        "column": "event_id"}},
            {"action_type": "patch_transformation",
             "params": {"step_id": "transform_conversions",
                        "patch_type": "strip_prefix",
                        "column": "campaign_id"}},
            {"action_type": "patch_transformation",
             "params": {"step_id": "transform_conversions",
                        "patch_type": "cast_column",
                        "column": "campaign_id"}},
            {"action_type": "alert_upstream_team",
             "params": {"team": "meta_ads_api_team",
                        "issue_description":
                            "Graph API outage: N/A impressions in ~8 rows"}},
        ],
        "must_alert_upstream": True,
        "upstream_team_to_alert": "meta_ads_api_team",
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