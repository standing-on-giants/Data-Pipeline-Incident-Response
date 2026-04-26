# Visual Checklist for Demo Video

Capture the following sequence of screen recordings, code snippets, and static images. Ensure text is highly legible (large font size in PyCharm/Terminal) for video output.

### Must-Haves
- [ ] **0:00–0:15 | Slack/Slack-like Alert vs. Terminal (Static/Animation)**
  - *Source:* Mockup or real screenshot of pipeline failure alerts alongside a terminal window parsing `PipelineObservation`.
  - *Description:* The hook comparing the manual on-call burden to the automated agent intake.

- [ ] **0:15–0:55 | DAG Visualization of Medium Task (Graphic/Web View)**
  - *Source:* Visual map derived from `make_medium_task()` in `src/tasks.py`.
  - *Description:* Shows data flowing from `raw_order_items` -> `transform_items` -> `aggregate_summary` to highlight structural awareness.

- [ ] **0:15–0:55 | Action Space Table (Screen/Graphics)**
  - *Source:* Derived from the `action_space` definition in `src/environment.py`.
  - *Description:* Bulleted/tabular showcase specifically emphasizing actions like `compare_schema`, `patch_transformation`, and `handle_drift`.

- [ ] **0:15–0:55 | Dynamic Schema Drift Popup (Animation)**
  - *Source:* Terminal log capturing `[Dynamic schema drift observed...]`.
  - *Description:* Emphasizes the world changing *while* the agent operates. 

- [ ] **0:55–1:15 | Reward Function Snippet (Code Screenshot)**
  - *Source:* The logic from `src/environment.py` demonstrating the shooting-blind penalty (line ~570 where `penalty = -0.5`). 
  - *Description:* Proves the environment is rigorous and prevents LLM gaming workflows.

- [ ] **0:55–1:15 | GRPO Training Reward Curve (PNG)**
  - *Source:* `/kaggle/working/grpo_training_curves.png` generated at the end of the Kaggle notebook.
  - *Description:* Shows the model climbing out of the heavy initial penalty valleys.

- [ ] **1:15–1:45 | Before & After Terminal Outputs (Screen Recording)**
  - *Source:* Output logs of the Base model failing vs. the GRPO checkpoint succeeding.
  - *Description:* Side-by-side terminal windows. Left shows hallucinated actions (`alert_owner`); right shows precise multi-step execution yielding `ALL ASSERTIONS PASSING`.

- [ ] **1:45–2:00 | Final Rankings Evaluation Table (Static/Terminal)**
  - *Source:* The stdout generated from the `evaluate_models.py` block showing SFT / GRPO / GRPO_BEST solve rates side-by-side.
  - *Description:* The frank wrap-up presenting numerical improvement transparently.
