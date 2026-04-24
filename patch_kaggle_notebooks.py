import json
import os

# ── Patch strings ──────────────────────────────────────────────────────────
RELOAD_SNIPPET = (
    "# Force-reload src modules so changes from git pull take effect\n"
    "import importlib, sys as _sys\n"
    "for _mod in list(_sys.modules.keys()):\n"
    "    if _mod.startswith('src'):\n"
    "        del _sys.modules[_mod]\n"
    "print('Repo ready:', os.getcwd())\n"
)
OLD_PRINT = "print('Repo ready:', os.getcwd())\n"

# Two spacing variants that appear across notebooks
ENV_PATTERNS = [
    (
        "    env = DataPipelineEnv(task_id=task_id, max_steps=max_steps)\n",
        (
            "    try:\n"
            "        env = DataPipelineEnv(task_id=task_id, max_steps=max_steps)\n"
            "    except TypeError:\n"
            "        env = DataPipelineEnv(task_id=task_id)\n"
            "        env.MAX_STEPS = max_steps\n"
        ),
    ),
    (
        "    env     = DataPipelineEnv(task_id=task_id, max_steps=max_steps)\n",
        (
            "    try:\n"
            "        env     = DataPipelineEnv(task_id=task_id, max_steps=max_steps)\n"
            "    except TypeError:\n"
            "        env     = DataPipelineEnv(task_id=task_id)\n"
            "        env.MAX_STEPS = max_steps\n"
        ),
    ),
]


def patch_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"]
        is_list = isinstance(src, list)
        lines = src if is_list else src.splitlines(keepends=True)

        full = "".join(lines)
        orig = full

        # Fix 1: importlib reload after git pull
        if "git pull" in full and OLD_PRINT in full and "Force-reload" not in full:
            full = full.replace(OLD_PRINT, RELOAD_SNIPPET)
            print(f"  [reload]  {os.path.basename(path)}")

        # Fix 2: defensive env creation
        for old_pat, new_pat in ENV_PATTERNS:
            if old_pat in full and "except TypeError" not in full:
                full = full.replace(old_pat, new_pat)
                print(f"  [env-fix] {os.path.basename(path)}")
                break

        if full != orig:
            cell["source"] = full.splitlines(keepends=True) if is_list else full
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        print(f"  -> saved  {os.path.basename(path)}")
    else:
        print(f"  -> ok (no changes needed): {os.path.basename(path)}")


NOTEBOOKS = [
    "run_on_kaggle/run_on_kaggle_LlaMa.ipynb",
    "run_on_kaggle/run_on_kaggle_qwen3vl.ipynb",
    "run_on_kaggle/run_on_kaggle_qwen_1.5b.ipynb",
    "run_on_kaggle/run_on_kaggle_qwen_fixed.ipynb",
    "run_on_kaggle/run_on_kaggle_qwen_new.ipynb",
    "run_on_kaggle/run_on_kaggle_qwen_v2.ipynb",
]

for nb_path in NOTEBOOKS:
    patch_notebook(nb_path)
