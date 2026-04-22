# Kept for backwards compatibility — core logic is now in eduautoml/automl.py
from eduautoml.automl import AutoML

def select_best_model(X, y, task="classification"):
    """Deprecated: Use AutoML().fit() instead."""
    raise DeprecationWarning("Use AutoML().fit(df, target) instead.")
