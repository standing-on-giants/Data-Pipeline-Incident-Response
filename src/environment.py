"""
DataPipelineEnv — the core RL environment.

Implements the OpenEnv interface:
  reset()  → PipelineObservation
  step()   → StepResult
  state()  → dict
"""
from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.models import (
    AssertionResult, DAGNode, HistoricalRun,
    PipelineAction, PipelineObservation, StepResult,
)
from src.assertions import check_assertion
from src.pipeline_runner import execute_pipeline
from src.tasks import get_task


class DataPipelineEnv:
    """
    Stateful environment for data pipeline incident response.

    Episode flow:
      1. reset(task_id)          → initial observation (assertions failing)
      2. agent calls read/check actions to diagnose
      3. agent applies fix actions (add_data_filter / patch_transformation)
      4. agent calls run_pipeline → new assertion results + reward
      5. done when all assertions pass OR max_steps reached
    """

    def __init__(self, task_id: str = "easy", max_steps: int = 20) -> None:
        self.task_id = task_id
        self.max_steps = max_steps
        self._task: Dict[str, Any] = {}
        self._raw_tables: Dict[str, pd.DataFrame] = {}
        self._dag: List[Dict[str, Any]] = []
        self._all_tables: Dict[str, pd.DataFrame] = {}

        # Tracking
        self._step_number: int = 0
        self._done: bool = False
        self._actions_taken: List[str] = []
        self._inspected_tables: Set[str] = set()   # for "shooting blind" penalty
        self._last_assertion_results: List[AssertionResult] = []
        self._reward_accumulator: float = 0.0

        # Carried between steps for reward delta computation
        self._prev_passed_ids: Set[str] = set()

        # Data last shown to agent via read actions
        self._last_data_sample: Optional[List[Dict[str, Any]]] = None
        self._last_schema: Optional[Dict[str, str]] = None
        self._last_hist_schema: Optional[Dict[str, str]] = None
        self._last_schema_diff: Optional[Dict[str, str]] = None
        self._last_action_result: str = ""
        self._pipeline_run_count: int = 0
        self._applied_drift_events: Set[str] = set()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self, task_id: Optional[str] = None) -> PipelineObservation:
        if task_id:
            self.task_id = task_id

        self._task        = get_task(self.task_id)
        self._raw_tables  = {k: v.copy() for k, v in self._task["raw_tables"].items()}
        self._dag         = copy.deepcopy(self._task["dag"])

        self._step_number        = 0
        self._done               = False
        self._actions_taken      = []
        self._inspected_tables   = set()
        self._reward_accumulator = 0.0
        self._last_data_sample   = None
        self._last_schema        = None
        self._last_hist_schema   = None
        self._last_schema_diff   = None
        self._last_action_result = "Pipeline reset. Initial assertion results loaded."
        self._pipeline_run_count = 0
        self._applied_drift_events = set()
        self._task["accepted_assertions"] = []
        self._task["alerts_sent"]         = []

        # Initial pipeline run (faults already injected)
        self._all_tables = execute_pipeline(self._raw_tables, self._dag)
        self._last_assertion_results = self._run_all_assertions()
        self._prev_passed_ids = {r.assertion_id for r in self._last_assertion_results if r.passed}

        return self._build_observation()

    def step(self, action: PipelineAction) -> StepResult:
        if self._done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0, done=True,
                info={"message": "Episode already complete."},
            )

        self._step_number += 1
        self._last_data_sample = None
        self._last_schema      = None
        self._last_hist_schema = None
        self._last_schema_diff = None

        reward, result_msg = self._dispatch_action(action)
        self._last_action_result = result_msg
        self._actions_taken.append(
            f"[{self._step_number}] {action.action_type}({action.params})"
        )
        self._reward_accumulator += reward

        # Check terminal condition
        all_passed    = all(r.passed for r in self._last_assertion_results)
        max_steps_hit = self._step_number >= self.max_steps
        self._done    = all_passed or max_steps_hit

        # Terminal bonus
        if all_passed:
            reward += 1.0
            self._reward_accumulator += 1.0
            self._last_action_result += " [PASSED] ALL ASSERTIONS PASSING — episode complete!"
        elif max_steps_hit and not all_passed:
            self._last_action_result += " [WARNING] Max steps reached."

        obs = self._build_observation()
        return StepResult(
            observation=obs, reward=reward, done=self._done,
            info={"total_reward": round(self._reward_accumulator, 3)},
        )

    def state(self) -> Dict[str, Any]:
        return {
            "task_id":     self.task_id,
            "step_number": self._step_number,
            "done":        self._done,
            "assertions_passed": sum(1 for r in self._last_assertion_results if r.passed),
            "assertions_total":  len(self._last_assertion_results),
            "total_reward": round(self._reward_accumulator, 3),
        }

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Action dispatcher
    # ------------------------------------------------------------------ #

    def _dispatch_action(
        self, action: PipelineAction
    ) -> Tuple[float, str]:
        p = action.params

        if action.action_type == "read_data_sample":
            return self._act_read_data_sample(
                table=p.get("table", ""),
                n_rows=int(p.get("n_rows", 20)),
                filter_col=p.get("filter_col"),
                filter_val=p.get("filter_val"),
            )

        elif action.action_type == "check_schema":
            return self._act_check_schema(table=p.get("table", ""))

        elif action.action_type == "compare_schema":
            return self._act_compare_schema(table=p.get("table", ""))

        elif action.action_type == "handle_drift":
            return self._act_handle_drift(
                strategy=p.get("strategy", "detect"),
                table=p.get("table", ""),
                step_id=p.get("step_id", ""),
                column=p.get("column", ""),
                filter_condition=p.get("filter_condition", ""),
                team=p.get("team", "meta_ads_api_team"),
                issue=p.get("issue_description", ""),
                old_column=p.get("old_column", "spend"),
                new_column=p.get("new_column", "total_spend"),
            )

        elif action.action_type == "run_quality_assertion":
            return self._act_run_assertion(assertion_id=p.get("assertion_id", ""))

        elif action.action_type == "add_data_filter":
            return self._act_add_filter(
                step_id=p.get("step_id", ""),
                filter_condition=p.get("filter_condition", ""),
            )

        elif action.action_type == "patch_transformation":
            return self._act_patch(
                step_id=p.get("step_id", ""),
                patch_type=p.get("patch_type", ""),
                column=p.get("column", ""),
                extra=p,
            )

        elif action.action_type == "backfill_partition":
            return self._act_backfill(date=p.get("date", ""))

        elif action.action_type == "alert_upstream_team":
            return self._act_alert(
                team=p.get("team", "unknown"),
                issue=p.get("issue_description", ""),
            )

        elif action.action_type == "mark_acceptable":
            return self._act_mark_acceptable(
                assertion_id=p.get("assertion_id", ""),
                reason=p.get("reason", ""),
            )

        elif action.action_type == "run_pipeline":
            return self._act_run_pipeline()

        else:
            return -0.1, f"Unknown action type: {action.action_type}"

    # ------------------------------------------------------------------ #
    # Individual action handlers
    # ------------------------------------------------------------------ #

    def _act_read_data_sample(
        self, table: str, n_rows: int, filter_col=None, filter_val=None
    ) -> Tuple[float, str]:
        df = self._all_tables.get(table)
        if df is None:
            df = self._raw_tables.get(table)
        if df is None:
            return -0.1, f"Table '{table}' not found."

        sample = df.head(n_rows)
        if filter_col and filter_col in df.columns:
            mask = df[filter_col].isna() if filter_val is None else (df[filter_col] == filter_val)
            sample = df[mask].head(n_rows)

        self._last_data_sample = sample.to_dict(orient="records")
        msg = (f"Showing {len(sample)} rows from '{table}' "
               f"({'filtered' if filter_col else 'unfiltered'}).")

        # Reward: 0 for the first inspection of this table; -0.05 for repeated reads
        reward = 0.0 if table not in self._inspected_tables else -0.05
        self._inspected_tables.add(table)
        return reward, msg

    def _act_check_schema(self, table: str) -> Tuple[float, str]:
        schemas = self._task.get("schemas", {})
        if table in schemas:
            self._last_schema = schemas[table]["current"]
            self._inspected_tables.add(table)
            return 0.0, f"Schema for '{table}' loaded."
        # Try to infer from raw table
        df = self._raw_tables.get(table)
        if df is not None:
            self._last_schema = {c: str(df[c].dtype) for c in df.columns}
            self._inspected_tables.add(table)
            return 0.0, f"Inferred schema for '{table}'."
        return -0.1, f"No schema info found for '{table}'."

    def _act_compare_schema(self, table: str) -> Tuple[float, str]:
        schemas = self._task.get("schemas", {})
        if table not in schemas:
            return -0.1, f"No historical schema for '{table}'."

        cur  = schemas[table]["current"]
        hist = schemas[table]["historical"]

        diff: Dict[str, str] = {}
        for col, dtype in cur.items():
            if col not in hist:
                diff[col] = f"NEW column (type: {dtype})"
            elif hist[col] != dtype:
                diff[col] = f"TYPE CHANGED: {hist[col]} → {dtype}"
        for col in hist:
            if col not in cur:
                diff[col] = "REMOVED column"

        self._last_schema      = cur
        self._last_hist_schema = hist
        self._last_schema_diff = diff if diff else {"info": "No schema changes detected."}
        self._inspected_tables.add(table)
        return 0.0, f"Schema diff for '{table}' loaded. {len(diff)} change(s) detected."

    def _act_run_assertion(self, assertion_id: str) -> Tuple[float, str]:
        assertion = next(
            (a for a in self._task["assertions"] if a["id"] == assertion_id), None
        )
        if not assertion:
            return -0.1, f"Assertion '{assertion_id}' not found."
        result = check_assertion(self._all_tables, assertion)
        # Update in-place so the observation stays fresh
        self._last_assertion_results = [
            result if r.assertion_id == assertion_id else r
            for r in self._last_assertion_results
        ]
        status = "PASSED" if result.passed else "FAILED"
        return 0.0, (f"Assertion {assertion_id} re-run: {status}. "
                     f"{result.actual}")

    def _act_handle_drift(
        self,
        strategy: str,
        table: str,
        step_id: str,
        column: str,
        filter_condition: str,
        team: str,
        issue: str,
        old_column: str,
        new_column: str,
    ) -> Tuple[float, str]:
        s = (strategy or "detect").strip().lower()

        if s == "detect":
            target_table = table or "raw_ads_insights"
            return self._act_compare_schema(target_table)

        if s == "numeric_format":
            return self._act_patch(
                step_id=step_id or "transform_insights",
                patch_type="parse_currency",
                column=column or "spend",
                extra={},
            )

        if s == "null_fill":
            return self._act_patch(
                step_id=step_id or "transform_insights",
                patch_type="coalesce",
                column=column or "spend",
                extra={},
            )

        if s == "type_cast":
            return self._act_patch(
                step_id=step_id or "transform_conversions",
                patch_type="cast_column",
                column=column or "campaign_id",
                extra={},
            )

        if s == "join_key_prefix":
            return self._act_patch(
                step_id=step_id or "transform_conversions",
                patch_type="strip_prefix",
                column=column or "campaign_id",
                extra={},
            )

        if s == "filter_invalid":
            return self._act_add_filter(
                step_id=step_id or "transform_insights",
                filter_condition=filter_condition or f"{(column or 'impressions')} IS NOT NULL",
            )

        if s == "alert_upstream":
            return self._act_alert(
                team=team or "meta_ads_api_team",
                issue=issue or "Schema drift detected in upstream API payload",
            )

        if s in ("resolve_column_rename", "column_rename"):
            target_table = table or "raw_ads_insights"
            src_col = old_column or "spend"
            dst_col = new_column or "total_spend"
            df = self._raw_tables.get(target_table)
            if df is None:
                return -0.1, f"Table '{target_table}' not found for drift resolution."
            if src_col in df.columns:
                return 0.0, f"Column '{src_col}' already present in '{target_table}'."
            if dst_col not in df.columns:
                return -0.1, (
                    f"Cannot resolve rename drift: '{dst_col}' missing in '{target_table}'."
                )

            # Compatibility shim for renamed upstream columns.
            df[src_col] = df[dst_col]
            self._raw_tables[target_table] = df

            schemas = self._task.get("schemas", {})
            cur_schema = schemas.get(target_table, {}).get("current")
            if isinstance(cur_schema, dict) and dst_col in cur_schema and src_col not in cur_schema:
                cur_schema[src_col] = cur_schema[dst_col]

            return 0.2, (
                f"Resolved schema rename drift in '{target_table}': mirrored "
                f"'{dst_col}' to '{src_col}'."
            )

        return -0.1, f"Unknown handle_drift strategy '{strategy}'."

    def _act_add_filter(
        self, step_id: str, filter_condition: str
    ) -> Tuple[float, str]:
        step = self._find_step(step_id)
        if step is None:
            return -0.1, f"Step '{step_id}' not found in DAG."

        # Shooting-blind penalty: modified a step without inspecting its table
        penalty = 0.0
        if step["input_table"] not in self._inspected_tables:
            penalty = -0.5

        step["applied_filters"].append(filter_condition)
        msg = f"Filter '{filter_condition}' added to step '{step_id}'."
        if penalty < 0:
            msg += " [PENALTY]: filter applied without reading the data first."
        return penalty, msg

    def _act_patch(
        self, step_id: str, patch_type: str, column: str, extra: Dict
    ) -> Tuple[float, str]:
        step = self._find_step(step_id)
        if step is None:
            return -0.1, f"Step '{step_id}' not found in DAG."

        penalty = 0.0
        if step["input_table"] not in self._inspected_tables:
            penalty = -0.5

        patch = {
            "patch_type":    patch_type,
            "column":        column,
            "default_value": extra.get("default_value"),
            "target_type":   extra.get("target_type"),
        }
        step["applied_patches"].append(patch)
        msg = f"Patch '{patch_type}' on column '{column}' applied to step '{step_id}'."
        if penalty < 0:
            msg += " [PENALTY]: patch applied without reading the data first."
        return penalty, msg

    def _act_backfill(self, date: str) -> Tuple[float, str]:
        # In our simulation, backfill just re-runs the full pipeline
        # (same as run_pipeline but scoped to a date — we treat it identically)
        return self._act_run_pipeline()

    def _act_alert(self, team: str, issue: str) -> Tuple[float, str]:
        self._task["alerts_sent"].append({"team": team, "issue": issue})
        expected_team = self._task.get("upstream_team_to_alert")
        must_alert    = self._task.get("must_alert_upstream", False)

        if must_alert and expected_team and team == expected_team:
            return 0.5, f"[SUCCESS] Correct team alerted: '{team}'. Issue recorded."
        elif not must_alert:
            return -0.2, f"Alert sent to '{team}' but escalation was not needed."
        else:
            return 0.0, f"Alert sent to '{team}'. (Correct team is '{expected_team}'.)"

    def _act_mark_acceptable(
        self, assertion_id: str, reason: str
    ) -> Tuple[float, str]:
        assertion = next(
            (a for a in self._task["assertions"] if a["id"] == assertion_id), None
        )
        if not assertion:
            return -0.1, f"Assertion '{assertion_id}' not found."

        # Is this assertion genuinely acceptable (i.e., has a fixable-only scope)?
        # For hard task: C5 (red herring — always passes) is never acceptable.
        # Marking a failing critical assertion as acceptable is a -1.0 penalty.
        result = next(
            (r for r in self._last_assertion_results if r.assertion_id == assertion_id),
            None,
        )
        if result and not result.passed:
            self._task["accepted_assertions"].append(assertion_id)
            return -1.0, (f"[FAILED] Swept under rug: assertion {assertion_id} is still failing "
                          f"and was marked acceptable. Penalty applied.")
        elif result and result.passed:
            return -0.1, f"Assertion {assertion_id} is already passing — no need to mark it."
        return -0.1, f"Could not evaluate assertion {assertion_id}."

    def _act_run_pipeline(self) -> Tuple[float, str]:
        self._pipeline_run_count += 1
        drift_messages = self._apply_scheduled_drift(self._pipeline_run_count)

        # Re-execute pipeline with current dag state
        self._all_tables = execute_pipeline(self._raw_tables, self._dag)
        new_results      = self._run_all_assertions()

        new_passed_ids  = {r.assertion_id for r in new_results if r.passed}
        prev_passed_ids = self._prev_passed_ids

        gained  = new_passed_ids - prev_passed_ids
        lost    = prev_passed_ids - new_passed_ids

        reward  = len(gained) * 0.4 - len(lost) * 0.5

        self._last_assertion_results = new_results
        self._prev_passed_ids        = new_passed_ids

        n_pass = len(new_passed_ids)
        n_tot  = len(new_results)
        msg = (f"Pipeline re-run: {n_pass}/{n_tot} assertions passing. "
               f"+{len(gained)} gained, -{len(lost)} lost.")
        if drift_messages:
            msg += " Drift events applied: " + " | ".join(drift_messages)
        return reward, msg

    def _apply_scheduled_drift(self, run_index: int) -> List[str]:
        schedule = self._task.get("drift_schedule", [])
        if not schedule:
            return []

        messages: List[str] = []
        for i, event in enumerate(schedule):
            if int(event.get("run_index", -1)) != run_index:
                continue

            event_id = str(event.get("id", f"evt_{run_index}_{i}"))
            if event_id in self._applied_drift_events:
                continue

            event_type = str(event.get("type", "")).strip().lower()

            if event_type == "rename_column":
                table = event.get("table", "")
                old_col = event.get("from", "")
                new_col = event.get("to", "")
                df = self._raw_tables.get(table)
                if df is not None and old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
                    self._raw_tables[table] = df

                    schemas = self._task.get("schemas", {})
                    cur_schema = schemas.get(table, {}).get("current")
                    if isinstance(cur_schema, dict) and old_col in cur_schema:
                        old_dtype = cur_schema.pop(old_col)
                        cur_schema[new_col] = old_dtype

                    messages.append(f"{table}.{old_col} renamed to {new_col}")
                else:
                    messages.append(f"rename_column skipped for {table}.{old_col}")

            elif event_type == "auth_format":
                self._task["current_auth_format"] = event.get("format", "unknown")
                messages.append(
                    f"auth format rotated to {self._task['current_auth_format']}"
                )

            elif event_type == "rate_limit":
                self._task["current_rate_limit"] = int(event.get("max_calls", 1))
                messages.append(
                    f"rate limit tightened to {self._task['current_rate_limit']} calls/window"
                )

            else:
                messages.append(f"unknown drift event type '{event_type}' ignored")

            self._applied_drift_events.add(event_id)

        return messages

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _find_step(self, step_id: str) -> Optional[Dict[str, Any]]:
        return next((s for s in self._dag if s["step_id"] == step_id), None)

    def _run_all_assertions(self) -> List[AssertionResult]:
        results = []
        for a in self._task["assertions"]:
            results.append(check_assertion(self._all_tables, a))
        return results

    def _build_observation(self) -> PipelineObservation:
        failed  = [r for r in self._last_assertion_results if not r.passed]
        passed  = [r for r in self._last_assertion_results if r.passed]
        all_ok  = len(failed) == 0

        dag_nodes = [
            DAGNode(
                step_id=s["step_id"],
                input_table=s["input_table"],
                output_table=s["output_table"],
                transformation_description=s["transformation_description"],
                applied_filters=list(s.get("applied_filters", [])),
                applied_patches=[
                    f"{p.get('patch_type')}({p.get('column')})"
                    for p in s.get("applied_patches", [])
                ],
            )
            for s in self._dag
        ]

        hist_runs = [
            HistoricalRun(**r) for r in self._task.get("historical_runs", [])
        ]

        return PipelineObservation(
            task_id=self.task_id,
            difficulty=self._task.get("difficulty", ""),
            description=self._task.get("description", ""),
            step_number=self._step_number,
            max_steps=self.max_steps,
            dag_structure=dag_nodes,
            failed_assertions=failed,
            passed_assertions=passed,
            historical_runs=hist_runs,
            data_sample=self._last_data_sample,
            current_schema=self._last_schema,
            historical_schema=self._last_hist_schema,
            schema_diff=self._last_schema_diff,
            last_action_result=self._last_action_result,
            actions_taken=list(self._actions_taken),
            pipeline_passed=all_ok,
            alert_sent=len(self._task.get("alerts_sent", [])) > 0,
        )