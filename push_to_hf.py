"""
push_to_hf.py  —  Upload the Data Pipeline Incident Response env to a HuggingFace repo.
"""

# ============================================================
# MACROS — edit these
# ============================================================

HF_TOKEN      = "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"
HF_USERNAME   = "Abhinav-hf"
REPO_NAME     = "data-pipeline-incident-qwen-grpo"
REPO_TYPE     = "space"   # keep this

LOCAL_SRC_DIR = r"C:\Users\Pratham\OneDrive\Desktop\RL_Data_Pipeline agents\Meta_hackathon"

# ============================================================
# FILES TO PUSH
# ============================================================

FILES_TO_PUSH = [
    # --- core env ---
    ("src/__init__.py",          "src/__init__.py"),
    ("src/environment.py",       "src/environment.py"),
    ("src/models.py",            "src/models.py"),
    ("src/assertions.py",        "src/assertions.py"),
    ("src/pipeline_runner.py",   "src/pipeline_runner.py"),
    ("src/tasks.py",             "src/tasks.py"),

    # --- server ---
    ("server/app.py",            "server/app.py"),
    ("server/__init__.py",       "server/__init__.py"),

    # --- infra (IMPORTANT for Docker Space) ---
    ("Dockerfile",               "Dockerfile"),
    ("requirements.txt",         "requirements.txt"),
    ("README.md",                "README.md"),

    # --- notebook ---
    ("run_on_kaggle_qwen_new.ipynb", "run_on_kaggle_qwen_new.ipynb"),
]

# ============================================================
# SCRIPT
# ============================================================

import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file

def main():

    # --- validate ---
    root = Path(LOCAL_SRC_DIR).expanduser().resolve()
    if not root.exists():
        print(f"[ERROR] LOCAL_SRC_DIR does not exist: {root}")
        sys.exit(1)

    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    api = HfApi(token=HF_TOKEN)

    # ============================================================
    # CREATE REPO
    # ============================================================
    print(f"[1/3] Ensuring repo exists: {repo_id} (type={REPO_TYPE})")

    if REPO_TYPE == "space":
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            token=HF_TOKEN,
            space_sdk="docker",   # ✅ FIXED
            exist_ok=True,
            private=False,
        )
    else:
        create_repo(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            token=HF_TOKEN,
            exist_ok=True,
            private=False,
        )

    print("      Repo ready.")

    # ============================================================
    # UPLOAD FILES
    # ============================================================
    print(f"[2/3] Uploading files...")

    skipped = []
    uploaded = []

    for local_rel, remote_path in FILES_TO_PUSH:
        local_abs = root / local_rel

        if not local_abs.exists():
            print(f"[SKIP] {local_rel}")
            skipped.append(local_rel)
            continue

        upload_file(
            path_or_fileobj=str(local_abs),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            token=HF_TOKEN,
            commit_message=f"upload {remote_path}",
        )

        print(f"[OK]   {local_rel}")
        uploaded.append(remote_path)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n[3/3] Done")
    print(f"Uploaded: {len(uploaded)}")
    print(f"Skipped : {len(skipped)}")

    print(f"\n👉 https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()