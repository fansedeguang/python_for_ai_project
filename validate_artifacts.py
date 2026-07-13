"""
validate_artifacts.py - Pre-submission schema validator

This script checks that the required CSV artifacts have:
- Correct column names (exact match)
- Valid enumerated values (e.g., 'disposition' column)
- The expected file existence

Run this script before final submission to avoid format-related penalties.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# =============================================================================
# CONFIGURATION: Expected schemas
# =============================================================================

EXPECTED_SCHEMAS: Dict[str, Dict] = {
    "heldout_predictions.csv": {
        "required_columns": ["id", "y_true", "y_pred", "score", "model_name", "ticket"],
        "allow_extra_columns": False,  # strict mode
        "non_null_columns": ["id", "y_true", "y_pred", "model_name", "ticket"],
        "id_column": "id",
        "validations": None,  # no enumeration check
    },
    "results/summary.csv": {
        "required_columns": [
            "ticket",
            "model_name",
            "dev_f1_target_1",
            "heldout_f1_target_1",
            "heldout_accuracy",
            "fixed_fp",
            "fixed_fn",
            "new_fp",
            "new_fn",
            "decision",
            "decision_reason",
        ],
        "allow_extra_columns": False,
        "non_null_columns": ["ticket", "model_name", "decision"],
        "id_column": None,
        "validations": None,
    },
    "results/threshold_sweep.csv": {
        "required_columns": ["ticket", "threshold", "precision_target_1", "recall_target_1", "f1_target_1"],
        "allow_extra_columns": False,
        "non_null_columns": ["ticket", "threshold", "f1_target_1"],
        "id_column": None,
        "validations": {
            "threshold": lambda s: (s >= 0).all() and (s <= 1).all(),  # between 0 and 1
        },
    },
    "results/data_quality_audit.csv": {
        "required_columns": ["id", "issue_type", "evidence", "disposition", "confidence"],
        "allow_extra_columns": False,
        "non_null_columns": ["id", "disposition"],
        "id_column": "id",
        "validations": {
            "disposition": lambda s: s.isin(
                ["fix", "keep_but_flag", "ambiguous", "reject_false_positive"]
            ).all(),
        },
    },
}

EXPECTED_FILES = list(EXPECTED_SCHEMAS.keys())


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_file_exists(path: Path) -> bool:
    """Return True if file exists and is non-empty."""
    if not path.exists():
        print(f"❌ File missing: {path}")
        return False
    if path.stat().st_size == 0:
        print(f"❌ File is empty: {path}")
        return False
    return True


def validate_schema(path: Path, schema: Dict) -> Tuple[bool, List[str]]:
    """
    Validate a single CSV file against its schema.
    Returns (is_valid, list_of_error_messages).
    """
    errors = []
    try:
        df = pd.read_csv(path)
    except Exception as e:
        errors.append(f"Could not read CSV: {e}")
        return False, errors

    # 1. Check required columns
    required = set(schema["required_columns"])
    present = set(df.columns)
    missing = required - present
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")

    if not schema["allow_extra_columns"]:
        extra = present - required
        if extra:
            errors.append(f"Extra columns (not allowed): {sorted(extra)}")

    # 2. Check non-null constraints
    for col in schema.get("non_null_columns", []):
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' has {null_count} null values (must be non-null).")

    # 3. Custom validations (e.g., enumeration, range)
    for col, validator in schema.get("validations", {}).items():
        if col not in df.columns:
            continue  # skip if column missing (already reported above)
        if not validator(df[col]):
            errors.append(f"Column '{col}' failed custom validation (e.g., invalid enumeration/range).")

    # 4. Optional: Check ID type for heldout_predictions (should be integer/string)
    id_col = schema.get("id_column")
    if id_col and id_col in df.columns:
        # For data_quality_audit, we could check IDs are in the heldout set, but we don't have the split here.
        # We skip to keep it simple; but we can warn.
        if df[id_col].dtype == 'object':
            # If it's string, might be okay; but we could suggest conversion
            pass

    return len(errors) == 0, errors


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("🔍 Starting artifact validation...\n")
    project_root = Path.cwd()
    all_passed = True

    for rel_path, schema in EXPECTED_SCHEMAS.items():
        full_path = project_root / rel_path
        print(f"📄 Checking: {rel_path}")

        # Step 1: File existence
        if not check_file_exists(full_path):
            all_passed = False
            print("   ❌ Skipping further checks (file missing).\n")
            continue

        # Step 2: Schema validation
        valid, errors = validate_schema(full_path, schema)
        if valid:
            print("   ✅ Schema valid.\n")
        else:
            all_passed = False
            print("   ❌ Schema errors:")
            for err in errors:
                print(f"      - {err}")
            print("")

    # Summary
    print("=" * 50)
    if all_passed:
        print("✅ All required artifacts are valid and ready for submission!")
        sys.exit(0)
    else:
        print("❌ Some artifacts have issues. Please fix them before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    main()