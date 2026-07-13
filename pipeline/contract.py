"""
pipeline/contract.py - Baseline Contract Wrapper

This module loads the reference contract from configs/project_contract.json
and provides:
- Baseline hyperparameters (TF-IDF + LR)
- Tolerance-aware metric validation
- Global random seed for reproducibility
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Global random seed (used across all modules)
DEFAULT_SEED = 42

# Path to the contract file (relative to project root)
CONTRACT_PATH = Path(__file__).parent.parent / "configs" / "project_contract.json"


def load_contract() -> Dict[str, Any]:
    """
    Load the reference contract from the JSON file.
    
    Returns:
        Dict containing the reference metrics and tolerance.
    
    Raises:
        FileNotFoundError: If contract JSON is missing.
        json.JSONDecodeError: If JSON is malformed.
    """
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Contract file not found at {CONTRACT_PATH}. "
            "Please ensure configs/project_contract.json exists."
        )
    
    with open(CONTRACT_PATH, "r", encoding="utf-8") as f:
        contract = json.load(f)
    
    # Basic validation: ensure required keys exist
    if "baseline_heldout_f1" not in contract and "reference_baseline_f1" not in contract:
        raise KeyError(
            "Contract missing required key: 'baseline_heldout_f1' or 'reference_baseline_f1'. "
            f"Found keys: {list(contract.keys())}"
        )
    if "tolerance" not in contract:
        raise KeyError(
            f"Contract missing required key: 'tolerance'. Found keys: {list(contract.keys())}"
        )

    return contract


def get_baseline_params() -> Dict[str, Any]:
    """
    Return the standard hyperparameters for the TF-IDF + Logistic Regression baseline.

    These values are chosen to match the course reference implementation.
    If the contract file includes them, they are loaded; otherwise, fallback defaults.
    """
    try:
        contract = load_contract()
        if "baseline_params" in contract:
            baseline_params = contract["baseline_params"].copy()
            if "tfidf" in baseline_params:
                tfidf_params = baseline_params["tfidf"].copy()
                if "ngram_range" in tfidf_params and isinstance(tfidf_params["ngram_range"], list):
                    tfidf_params["ngram_range"] = tuple(tfidf_params["ngram_range"])
                baseline_params["tfidf"] = tfidf_params
            return baseline_params
    except (FileNotFoundError, KeyError):
        # Fallback defaults that are known to reproduce the reference F1
        pass

    # These defaults are selected for reproducibility with sklearn 1.9.0
    return {
        "tfidf": {
            "max_features": 5000,
            "ngram_range": (1, 1),
            "lowercase": True,
            "stop_words": None,
            "use_idf": True,
            "smooth_idf": True,
            "sublinear_tf": True,
            "strip_accents": "unicode",
            "token_pattern": r"(?u)\b\w+\b",
        },
        "logistic_regression": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": DEFAULT_SEED,
            "class_weight": None,  # balanced will be explored in Ticket 4
        },
    }


def get_reference_metric() -> float:
    """Return the reference held-out F1 (target=1) from the contract."""
    contract = load_contract()
    return contract.get("baseline_heldout_f1", contract.get("reference_baseline_f1"))


def get_tolerance() -> float:
    """Return the allowed tolerance (e.g., 0.01) for baseline reproduction."""
    contract = load_contract()
    return contract.get("tolerance", 0.01)  # default 0.01 if missing


def check_baseline_match(
    computed_f1: float,
    metric_name: str = "heldout_f1_target_1",
    raise_on_fail: bool = True,
) -> bool:
    """
    Assert whether the computed F1 falls within the contract's tolerance.
    
    Args:
        computed_f1: The F1 score you computed on held-out (or dev) set.
        metric_name: Only used for error messages (e.g., "dev_f1" vs "heldout_f1").
        raise_on_fail: If True, raises RuntimeError; if False, returns bool.
    
    Returns:
        True if within tolerance, False otherwise (only if raise_on_fail=False).
    
    Raises:
        RuntimeError: If mismatch is detected and raise_on_fail=True.
    """
    reference = get_reference_metric()
    tolerance = get_tolerance()
    diff = abs(computed_f1 - reference)
    
    is_match = diff <= tolerance
    
    if not is_match and raise_on_fail:
        raise RuntimeError(
            f"❌ Baseline reproduction FAILED for {metric_name}.\n"
            f"   Computed: {computed_f1:.6f}, Reference: {reference:.6f}, "
            f"Diff: {diff:.6f} (tolerance: {tolerance:.4f})\n"
            "   This suggests split, preprocessing, or sklearn version mismatch. "
            "Check your pipeline and rerun with locked dependencies."
        )
    
    return is_match


def get_seed() -> int:
    """Return the global random seed."""
    return DEFAULT_SEED


# Optional: Module-level self-test (run with `python -m pipeline.contract`)
if __name__ == "__main__":
    print("🔍 Testing contract module...")
    try:
        reference = get_reference_metric()
        print(f"✅ Contract loaded: reference F1 = {reference:.4f}")
        print(f"✅ Tolerance = {get_tolerance()}")
        
        params = get_baseline_params()
        print(f"✅ Baseline params loaded: {params.keys()}")
        
        print("✅ All contract functions are working correctly.")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)