"""
pipeline/utils.py - Shared utilities for reproducibility
"""

import random
import numpy as np
from pipeline.contract import get_seed


def set_global_seed(seed: int = None):
    """
    Set the global random seed for Python's random, NumPy, and sklearn.
    
    Args:
        seed: If None, uses the seed from contract.py (DEFAULT_SEED = 42).
    """
    if seed is None:
        seed = get_seed()
    
    random.seed(seed)
    np.random.seed(seed)
    
    # sklearn's random_state is set per estimator, not globally.
    # But we set it here for any sklearn functions that use global randomness.
    try:
        import sklearn.utils
        sklearn.utils.check_random_state(seed)
    except ImportError:
        pass
    
    print(f"🌱 Global seed set to: {seed}")