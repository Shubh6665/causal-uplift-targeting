import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, average_precision_score, brier_score_loss
import logging

logger = logging.getLogger(__name__)

def compute_logloss(Y_true: np.ndarray, Y_prob: np.ndarray) -> float:
    """Primary factual metric for classification. Lower is better."""
    return float(log_loss(Y_true, Y_prob))

def compute_pr_auc(Y_true: np.ndarray, Y_prob: np.ndarray) -> float:
    """Secondary factual metric. Useful for highly imbalanced conversions (e.g. 0.29%)."""
    return float(average_precision_score(Y_true, Y_prob))

def compute_brier_score(Y_true: np.ndarray, Y_prob: np.ndarray) -> float:
    """Measures calibration of probabilities. Lower is better."""
    return float(brier_score_loss(Y_true, Y_prob))

def compute_factual_metrics(X: pd.DataFrame, T: np.ndarray, Y: np.ndarray, learner) -> dict:
    """
    Computes factual outcome prediction performance.
    Valid for S-Learner and T-Learner which predict factual outcomes directly.
    Not valid for X-Learner since its primary output is the effect tau_hat.
    """
    if learner.__class__.__name__ == 'XLearner':
        logger.warning("Factual metrics requested for X-Learner. X-Learner does not predict factuals in a single step like S/T learners.")
        raise ValueError("XLearner cannot directly predict factual outcomes in the same way as S/T learners.")
        
    try:
        Y_prob = learner.predict_factual(X, T)
    except AttributeError:
        raise AttributeError(f"Learner {learner.__class__.__name__} does not implement predict_factual.")
        
    metrics = {
        "logloss": compute_logloss(Y, Y_prob),
        "pr_auc": compute_pr_auc(Y, Y_prob),
        "brier_score": compute_brier_score(Y, Y_prob)
    }
    
    return metrics
