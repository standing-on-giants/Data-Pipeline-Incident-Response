"""
Pydantic models for the Data Pipeline Incident Response environment.
Compliant with the OpenEnv spec: typed Observation, Action, and Reward models.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class AssertionResult(BaseModel):
    assertion_id: str
    table: str
    assertion_type: str          # not_null | unique | row_count | value_range | type_check
    column: Optional[str] = None
    expected: str
    actual: str
    passed: bool
    failing_row_count: int = 0


class DAGNode(BaseModel):
    step_id: str
    input_table: str
    output_table: str
    transformation_description: str
    applied_filters: List[str] = Field(default_factory=list)
    applied_patches: List[str] = Field(default_factory=list)


class HistoricalRun(BaseModel):
    date: str
    status: str          # passed | failed
    row_count: int
    duration_s: int


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class PipelineObservation(BaseModel):
    """Everything the agent can see about the current environment state."""

    task_id: str
    difficulty: str
    description: str
    step_number: int
    max_steps: int

    dag_structure: List[DAGNode]
    failed_assertions: List[AssertionResult]
    passed_assertions: List[AssertionResult]
    historical_runs: List[HistoricalRun]

    # Populated only after the agent calls specific read actions
    data_sample: Optional[List[Dict[str, Any]]] = None
    current_schema: Optional[Dict[str, str]] = None
    historical_schema: Optional[Dict[str, str]] = None
    schema_diff: Optional[Dict[str, str]] = None    # new / removed / changed columns

    last_action_result: str
    actions_taken: List[str]
    pipeline_passed: bool
    alert_sent: bool


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class PipelineAction(BaseModel):
    """Actions the agent can take inside the environment."""

    action_type: Literal[
        "read_data_sample",       # look at rows in a table
        "check_schema",           # inspect current column types
        "compare_schema",         # diff current vs historical schema
        "run_quality_assertion",  # re-run a specific assertion on demand
        "add_data_filter",        # add a WHERE-style filter to a pipeline step
        "patch_transformation",   # apply a column-level fix (cast, coalesce, dedup)
        "backfill_partition",     # re-run pipeline for a specific date partition
        "alert_upstream_team",    # escalate an issue to the data source owner
        "mark_acceptable",        # consciously accept a known data quality issue
        "run_pipeline",           # re-execute full pipeline and see new assertion results
    ]
    params: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "examples": [
                {"action_type": "read_data_sample",
                 "params": {"table": "raw_orders", "n_rows": 20}},
                {"action_type": "add_data_filter",
                 "params": {"step_id": "transform_orders",
                            "filter_condition": "user_id IS NOT NULL"}},
                {"action_type": "patch_transformation",
                 "params": {"step_id": "transform_orders",
                            "patch_type": "cast_column",
                            "column": "revenue",
                            "target_type": "float"}},
                {"action_type": "run_pipeline", "params": {}},
            ]
        }


# ---------------------------------------------------------------------------
# Step result (returned by env.step())
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: PipelineObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)