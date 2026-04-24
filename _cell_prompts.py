SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data engineer diagnosing and fixing broken data pipelines.

You will receive the current state of a pipeline (failing assertions, DAG structure,
historical run info) and must choose ONE action to take each turn.

WORKFLOW (follow this order strictly):
1. FIRST: read_data_sample on the raw input table(s) to see what the data looks like.
2. THEN: Use check_schema or compare_schema if a type or schema issue is suspected.
3. If you see any schema drift signal (renamed/missing columns, changed types, auth format drift,
   or stricter rate-limit behavior), use handle_drift.
4. THEN: Apply the RIGHT fix using add_data_filter or patch_transformation.
5. THEN: Call run_pipeline to verify the fix.
6. ONLY AFTER fixing what you can: If some data is genuinely corrupted (e.g. "N/A" values
   that cannot be parsed), call alert_upstream_team.
7. If assertions are still failing after run_pipeline, investigate more and apply
   additional fixes. Do NOT just call run_pipeline again without changing something.

AVAILABLE ACTIONS (respond with ONLY a JSON object, no markdown, no prose):

{"action_type": "read_data_sample", "params": {"table": "<table_name>", "n_rows": 20}}
{"action_type": "check_schema", "params": {"table": "<table_name>"}}
{"action_type": "compare_schema", "params": {"table": "<table_name>"}}
{"action_type": "handle_drift", "params": {"strategy": "<detect|numeric_format|null_fill|type_cast|join_key_prefix|filter_invalid|resolve_column_rename|alert_upstream>", "table": "<table_name_optional>", "step_id": "<step_id_optional>", "column": "<column_optional>", "old_column": "<optional_old_name>", "new_column": "<optional_new_name>", "filter_condition": "<optional>", "team": "<optional>", "issue_description": "<optional>"}}
{"action_type": "run_quality_assertion", "params": {"assertion_id": "<e.g. A1>"}}
{"action_type": "add_data_filter", "params": {"step_id": "<step_id>", "filter_condition": "<e.g. user_id IS NOT NULL>"}}
{"action_type": "patch_transformation", "params": {"step_id": "<step_id>", "patch_type": "<cast_column|coalesce|dedup|parse_currency|strip_prefix>", "column": "<column_name>"}}
{"action_type": "backfill_partition", "params": {"date": "<YYYY-MM-DD>"}}
{"action_type": "alert_upstream_team", "params": {"team": "<team_name_snake_case>", "issue_description": "<short description>"}}
{"action_type": "mark_acceptable", "params": {"assertion_id": "<id>", "reason": "<reason>"}}
{"action_type": "run_pipeline", "params": {}}

KEY PATCH TYPES (you can chain multiple patches on the same step — they run in order):
- parse_currency: Use when a column has mixed formats like "$1,234.56" and "1234.56" and "N/A".
  It strips $, commas, casts to float, and converts N/A to NaN. Works on ANY column with "N/A" strings,
  not just currency — e.g. if a numeric column like impressions has "N/A" values, use parse_currency on it.
- coalesce: Use AFTER parse_currency to replace NaN/null with a default value (default is 0).
  IMPORTANT: After parse_currency, NaN values will cause value_range assertions to fail.
  You MUST chain a coalesce patch to fix this: {"action_type": "patch_transformation", "params": {"step_id": "<same_step>", "patch_type": "coalesce", "column": "<same_column>"}}
  If coalescing a denominator column (e.g. impressions used in CTR = clicks/impressions), coalescing to 0
  will cause division by zero. Instead, filter out those rows: add_data_filter with "column IS NOT NULL".
- cast_column: Use when a column needs simple numeric casting.
- dedup: Use when there are duplicate rows based on a key column.
  IMPORTANT: If a "unique" assertion is failing, the fix is ALWAYS dedup on the failing column.
  Do NOT use coalesce or add_data_filter for uniqueness failures — only dedup works.
- strip_prefix: Use when column values have a spurious prefix like "CMP_" that needs removal.
  Params: step_id, column. Optionally "prefix" (default "CMP_"). After stripping, chain cast_column
  if the underlying value should be numeric.

DRIFT HANDLING RULES:
- Use handle_drift when schema or contract changes between runs.
- handle_drift strategy mapping:
    detect -> compare_schema
    numeric_format -> patch_transformation(parse_currency)
    null_fill -> patch_transformation(coalesce)
    type_cast -> patch_transformation(cast_column)
    join_key_prefix -> patch_transformation(strip_prefix)
    filter_invalid -> add_data_filter
    resolve_column_rename -> restore compatibility for renamed columns (e.g. spend <- total_spend)
    alert_upstream -> alert_upstream_team
- For spend -> total_spend style drift, compare schema first, then patch transformations to align types.

UPSTREAM TEAM NAMING:
- Team names are always lowercase snake_case. Examples: meta_ads_api_team, data_engineering, vendor_support.
- If the description mentions "Meta", "Graph API", or "Meta Ads", the team is likely "meta_ads_api_team".

RULES:
- RESPOND WITH ONLY A JSON OBJECT. No markdown fences, no explanation, no prose.
- Do NOT call run_pipeline unless you applied a filter or patch since the last run.
- Do NOT apply a fix before reading the data — this will be penalised.
- NEVER use mark_acceptable. It always results in a heavy penalty. Instead, fix the data.
- After parse_currency, ALWAYS chain coalesce on the same column to handle NaN values before calling run_pipeline.
- If a "unique" assertion fails (e.g. uniqueness on order_item_id), the ONLY correct fix is dedup.
  Do NOT try coalesce, add_data_filter, or any other patch for uniqueness failures.
- If a computed column (like CTR) has a value_range failure, check ALL input columns in its formula.
  For example, if CTR = clicks/impressions and impressions has "N/A" strings, you must fix impressions
  with parse_currency first, then filter out null rows, before the computed column can produce valid values.
- If a joined output table has 0 rows (row_count assertion failing), the join keys likely don't match.
  Use compare_schema on the input tables to detect type/format drifts like string vs int, or unwanted
  prefixes on the join key. Apply strip_prefix + cast_column to align the keys.
- If pipeline_passed is true, you are done — unless the task description mentions alerting an upstream team.
- NEVER repeat the same action you already tried. If an action did not fix the problem, try a DIFFERENT action.
""").strip()


def _collect_schema_drift_signals(obs: PipelineObservation) -> List[str]:
    signals: List[str] = []
    desc = (obs.description or "").lower()
    if "schema drift" in desc or "contract" in desc:
        signals.append("Task description references schema/contract drift.")
    if obs.schema_diff:
        schema_diff_json = json.dumps(obs.schema_diff).lower()
        if "removed" in schema_diff_json:
            signals.append("Historical columns appear removed in current schema.")
        if "changed" in schema_diff_json:
            signals.append("Column types differ from historical schema.")
        if "new" in schema_diff_json:
            signals.append("New columns detected relative to historical schema.")
    for r in obs.failed_assertions:
        actual = (r.actual or "").lower()
        if "missing" in actual and "column" in actual:
            signals.append(f"Assertion {r.assertion_id} reports a missing column.")
        if "not found" in actual and "column" in actual:
            signals.append(f"Assertion {r.assertion_id} reports a renamed or deleted column.")
        if "type" in actual and ("object" in actual or "string" in actual):
            signals.append(f"Assertion {r.assertion_id} indicates possible type drift.")
    deduped: List[str] = []
    for s in signals:
        if s not in deduped:
            deduped.append(s)
    return deduped[:6]


def _detect_action_loop(actions_taken: List[str]) -> Optional[str]:
    # Detect if the last 3+ actions share the same action_type(params) pattern.
    # actions_taken format: "[step] action_type({'param': 'val'})"
    if len(actions_taken) < 3:
        return None

    def _extract_action_key(action_str: str) -> str:
        # Extract "action_type({params})" from "[N] action_type({params})"
        idx = action_str.find(']')
        if idx >= 0:
            return action_str[idx+1:].strip()
        return action_str.strip()

    last_3 = [_extract_action_key(a) for a in actions_taken[-3:]]
    if last_3[0] == last_3[1] == last_3[2]:
        return last_3[0]

    # Also detect 2-step oscillation: A, B, A, B
    if len(actions_taken) >= 4:
        last_4 = [_extract_action_key(a) for a in actions_taken[-4:]]
        if last_4[0] == last_4[2] and last_4[1] == last_4[3]:
            return f"{last_4[0]} and {last_4[1]} (oscillating)"

    return None


def _build_loop_hint(obs: PipelineObservation, looped_action: str) -> str:
    # Build a targeted hint based on what the model is stuck on and what assertions are failing
    hint = (
        f"\n[CRITICAL LOOP DETECTED]: You have been repeating '{looped_action}' "
        f"without making progress. This action is NOT solving the problem. "
        f"You MUST try a COMPLETELY DIFFERENT action type.\n"
    )

    # Analyze the failing assertions to suggest the right fix
    for r in obs.failed_assertions:
        atype = (r.assertion_type or "").lower()
        col = r.column or ""
        table = r.table or ""

        if atype == "unique":
            hint += (
                f"\n  -> Assertion {r.assertion_id} is a UNIQUENESS failure on column '{col}' "
                f"in table '{table}'. The ONLY fix for this is: "
                f'{{"action_type": "patch_transformation", "params": {{"step_id": "<find the step that outputs {table}>", "patch_type": "dedup", "column": "{col}"}}}}'
            )
        elif atype == "row_count":
            hint += (
                f"\n  -> Assertion {r.assertion_id} is a ROW COUNT failure on '{table}'. "
                f"This is usually caused by duplicate rows inflating the count. "
                f"Look for a uniqueness assertion on the same table and fix that with dedup first."
            )
        elif atype == "type_check":
            hint += (
                f"\n  -> Assertion {r.assertion_id} is a TYPE CHECK failure on '{col}' in '{table}'. "
                f"Use parse_currency on that column to convert string values to numeric."
            )
        elif atype == "value_range":
            hint += (
                f"\n  -> Assertion {r.assertion_id} is a VALUE RANGE failure on '{col}'. "
                f"Check if parse_currency was applied — if so, chain coalesce to replace NaN with 0."
            )

    # If the model is stuck on run_pipeline
    if "run_pipeline" in looped_action:
        hint += (
            "\n\n  You MUST apply a fix (patch_transformation or add_data_filter) "
            "BEFORE calling run_pipeline again. run_pipeline without changes does nothing."
        )

    return hint


def build_user_prompt(obs: PipelineObservation, step: int) -> str:
    failed_str = "\n".join(
        f"  [{r.assertion_id}] {r.assertion_type} on {r.table}"
        f"({r.column or 'N/A'}): {r.actual}"
        for r in obs.failed_assertions
    ) or "  (none -- all passing!)"

    passed_str = ", ".join(r.assertion_id for r in obs.passed_assertions) or "none"

    dag_str = "\n".join(
        f"  {n.step_id}: {n.input_table} -> {n.output_table}"
        + (f" | filters: {n.applied_filters}" if n.applied_filters else "")
        + (f" | patches: {n.applied_patches}" if n.applied_patches else "")
        for n in obs.dag_structure
    )

    hist_str = "\n".join(
        f"  {r.date}: {r.status} ({r.row_count} rows)"
        for r in obs.historical_runs
    )

    sample_str = ""
    if obs.data_sample:
        sample_rows = obs.data_sample[:5]
        null_rows = [r for r in obs.data_sample if any(v is None for v in r.values())]
        if null_rows:
            sample_str = (
                "\nDATA SAMPLE (first 5 rows):\n"
                + json.dumps(sample_rows, indent=2, default=str)
                + f"\nROWS WITH NULL VALUES ({len(null_rows)} found):\n"
                + json.dumps(null_rows[:5], indent=2, default=str)
            )
        else:
            sample_str = (
                "\nDATA SAMPLE (first 5 rows):\n"
                + json.dumps(sample_rows, indent=2, default=str)
            )

    schema_str = ""
    if obs.current_schema:
        schema_str = "\nCURRENT SCHEMA: " + json.dumps(obs.current_schema)
    if obs.schema_diff:
        schema_str += "\nSCHEMA DIFF vs HISTORICAL: " + json.dumps(obs.schema_diff)

    drift_signals = _collect_schema_drift_signals(obs)
    drift_str = ""
    if drift_signals:
        drift_str = "\nSCHEMA DRIFT SIGNALS:\n" + "\n".join(f"  - {s}" for s in drift_signals)

    actions_str = "\n".join(f"  {a}" for a in obs.actions_taken[-5:]) or "  (none yet)"

    read_actions  = sum(1 for a in obs.actions_taken if "read_data_sample" in a or "check_schema" in a)
    fix_actions   = sum(1 for a in obs.actions_taken if "add_data_filter" in a or "patch_transformation" in a)
    mark_actions  = sum(1 for a in obs.actions_taken if "mark_acceptable" in a)
    parse_done    = any("parse_currency" in a for a in obs.actions_taken)
    coalesce_done = any("coalesce" in a for a in obs.actions_taken)

    hint_str = ""

    # -- Loop detection (prompt-based, not harness override) --
    looped_action = _detect_action_loop(obs.actions_taken)
    if looped_action:
        hint_str += _build_loop_hint(obs, looped_action)
    else:
        # Standard hints (only if not already showing loop hint)
        if read_actions >= 2 and fix_actions == 0:
            hint_str = (
                "\n[HINT]: You have already read the data. "
                "Stop diagnosing. Apply a fix now using add_data_filter or patch_transformation, "
                "then call run_pipeline."
            )

    value_range_failing = any(
        r.assertion_type == "value_range" and "non-numeric" in r.actual
        for r in obs.failed_assertions
    )
    if parse_done and not coalesce_done and value_range_failing:
        hint_str += (
            "\n[CRITICAL]: A value_range assertion is STILL failing because parse_currency converts "
            "unparseable values (like 'N/A') to NaN, and NaN counts as out-of-range. "
            "You MUST apply a coalesce patch to replace NaN with 0 on the same column and step "
            "where you applied parse_currency."
        )
    if mark_actions >= 1:
        hint_str += (
            "\n[WARNING]: NEVER use mark_acceptable again. It gives a -1.0 penalty every time. "
            "Instead, apply a coalesce patch to fix NaN values, then run_pipeline."
        )
    recent = obs.actions_taken[-3:]
    recent_runs = sum(1 for a in recent if "run_pipeline" in a)
    if recent_runs >= 2 and not obs.pipeline_passed:
        hint_str += (
            "\n[CRITICAL]: You have called run_pipeline multiple times with no progress. "
            "You MUST apply a fix (patch_transformation or add_data_filter) before calling run_pipeline again."
        )

    return textwrap.dedent(f"""
    STEP {step}/{obs.max_steps}
    TASK: {obs.task_id} ({obs.difficulty})
    DESCRIPTION: {obs.description}
    PIPELINE PASSED: {obs.pipeline_passed}
    LAST ACTION RESULT: {obs.last_action_result}

    DAG STRUCTURE:
    {dag_str}

    FAILING ASSERTIONS:
    {failed_str}

    PASSING ASSERTIONS: {passed_str}

    HISTORICAL RUNS:
    {hist_str}

    RECENT ACTIONS TAKEN:
    {actions_str}
    {sample_str}{schema_str}{drift_str}{hint_str}

    Respond with exactly ONE action JSON object.
    """).strip()


print('Prompts ready.')
