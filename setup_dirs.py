import os
from pathlib import Path

# List of directories to create (relative to project root)
REQUIRED_DIRS = [
    "data",
    "configs",
    "pipeline",
    "experiments",
    "tickets",
    "predictions/dev_predictions",
    "results/figures",
    "logs",
    "tests",
]

# List of placeholder files to create (optional, to keep Git track)
PLACEHOLDER_FILES = [
    "pipeline/__init__.py",
    "pipeline/loader.py",
    "pipeline/preprocess.py",
    "pipeline/features.py",
    "pipeline/models.py",
    "pipeline/evaluate.py",
    "pipeline/utils.py",
    "experiments/run_baseline.py",
    "experiments/run_normalization.py",
    "experiments/run_shortcut_audit.py",
    "experiments/run_threshold_tuning.py",
    "experiments/run_data_audit.py",
    "experiments/run_final_heldout.py",
    "tickets/ticket-1-baseline.md",
    "tickets/ticket-2-normalization.md",
    "tickets/ticket-3-shortcuts.md",
    "tickets/ticket-4-decision-rule.md",
    "tickets/ticket-5-data-quality.md",
    "tests/test_split_integrity.py",
    "tests/test_artifact_schema.py",
    "tests/test_reproducibility.py",
    "logs/chat.md",
]


def create_directories():
    """Create all required directories if they do not exist."""
    print("📁 Creating project directories...")
    for dir_path in REQUIRED_DIRS:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}/")
    print("Directory creation complete.\n")


def create_placeholder_files():
    """
    Create empty placeholder files to keep the structure visible in Git.
    If a file already exists, it is left untouched.
    """
    print("📄 Creating placeholder files (if missing)...")
    for file_path in PLACEHOLDER_FILES:
        path = Path(file_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Placeholder - remove or modify as needed\n")
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⏭️  {file_path} already exists, skipped.")
    print("Placeholder file creation complete.")


def main():
    create_directories()
    create_placeholder_files()
    print("\n🎯 All set! You can now proceed with data download and development.")
    print("   Run: python download_data.py")


if __name__ == "__main__":
    main()