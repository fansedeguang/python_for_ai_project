"""
This module loads the raw train.csv and applies the fixed split from
split_indices.json. It ensures that train/dev/heldout IDs are mutually
exclusive and cover the entire dataset.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

# Paths relative to project root
DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_CSV_PATH = DATA_DIR / "train.csv"
SPLIT_JSON_PATH = DATA_DIR / "split_indices.json"


def load_split_indices() -> Dict[str, list]:
    """
    Load the fixed split indices from split_indices.json.
    
    Returns:
        Dict with keys: 'train_ids', 'dev_ids', 'heldout_ids'
    
    Raises:
        FileNotFoundError: If split_indices.json is missing.
    """
    if not SPLIT_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found at {SPLIT_JSON_PATH}. "
            "Please ensure data/split_indices.json exists."
        )
    
    with open(SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
        split_dict = json.load(f)
    
    # Validate that all required keys exist
    required_keys = ["train_ids", "dev_ids", "heldout_ids"]
    missing = [k for k in required_keys if k not in split_dict]
    if missing:
        raise KeyError(f"Split file missing keys: {missing}")
    
    return split_dict


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the raw CSV and split it into train, dev, and heldout DataFrames.
    
    Returns:
        Tuple of (train_df, dev_df, heldout_df), each with all original columns.
    
    Raises:
        FileNotFoundError: If train.csv is missing.
        ValueError: If IDs in splits don't match the CSV or overlap.
    """
    if not TRAIN_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_CSV_PATH}. "
            "Please run: python download_data.py"
        )
    
    # Load the full dataset
    df = pd.read_csv(TRAIN_CSV_PATH)
    
    # Ensure the 'id' column exists and is used as identifier
    if "id" not in df.columns:
        raise ValueError("CSV does not contain an 'id' column.")
    
    # Load split indices
    split_dict = load_split_indices()
    train_ids = set(split_dict["train_ids"])
    dev_ids = set(split_dict["dev_ids"])
    heldout_ids = set(split_dict["heldout_ids"])
    
    # Check for overlaps between splits (should be empty)
    overlap_train_dev = train_ids & dev_ids
    overlap_train_held = train_ids & heldout_ids
    overlap_dev_held = dev_ids & heldout_ids
    
    if overlap_train_dev or overlap_train_held or overlap_dev_held:
        raise ValueError(
            f"Overlap detected between splits:\n"
            f"  train ∩ dev: {overlap_train_dev}\n"
            f"  train ∩ heldout: {overlap_train_held}\n"
            f"  dev ∩ heldout: {overlap_dev_held}"
        )
    
    # Convert all IDs to same type (string or int) to avoid mismatch
    # Pandas reads IDs as int; JSON loads as int. We'll keep as int.
    df_ids = set(df["id"].tolist())
    all_split_ids = train_ids | dev_ids | heldout_ids
    
    # Check if any IDs in the split are missing from the CSV
    missing_from_csv = all_split_ids - df_ids
    if missing_from_csv:
        raise ValueError(
            f"Split IDs not found in CSV: {missing_from_csv}"
        )
    
    # Check if any IDs in the CSV are not assigned to any split
    unassigned = df_ids - all_split_ids
    if unassigned:
        raise ValueError(
            f"Unassigned IDs found in CSV (not in any split): {unassigned}"
        )
    
    # Perform the split using boolean indexing
    train_df = df[df["id"].isin(train_ids)].reset_index(drop=True)
    dev_df = df[df["id"].isin(dev_ids)].reset_index(drop=True)
    heldout_df = df[df["id"].isin(heldout_ids)].reset_index(drop=True)
    
    return train_df, dev_df, heldout_df


def get_X_y_from_df(df: pd.DataFrame, text_col: str = "text", target_col: str = "target"):
    """
    Convenience function: extract features (text) and labels from a DataFrame.
    
    Args:
        df: DataFrame from load_data() (train_df, dev_df, or heldout_df).
        text_col: Name of the text column (default: "text").
        target_col: Name of the target column (default: "target").
    
    Returns:
        Tuple of (X: Series of texts, y: Series of labels).
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")
    
    return df[text_col], df[target_col]


# Self-test: run `python -m pipeline.loader` to verify the split
if __name__ == "__main__":
    print("🔍 Testing data loader...")
    try:
        train_df, dev_df, heldout_df = load_data()
        print(f"🔍 Sample raw text: {repr(train_df['text'].iloc[0])}")
        print(f"✅ Loaded successfully:")
        print(f"   Train: {len(train_df)} samples")
        print(f"   Dev:   {len(dev_df)} samples")
        print(f"   Heldout: {len(heldout_df)} samples")
        
        # Quick sanity check on target distribution
        print(f"\n📊 Target distribution (train):")
        print(train_df["target"].value_counts().to_dict())
        
        # Verify the split is exhaustive
        total = len(train_df) + len(dev_df) + len(heldout_df)
        print(f"\n✅ Total samples: {total}")
        
    except Exception as e:
        print(f"❌ Error: {e}")