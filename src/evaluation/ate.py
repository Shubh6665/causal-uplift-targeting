import numpy as np

def mean_predicted_effect(tau_hat: np.ndarray) -> float:
    """
    Average of estimated conditional treatment effects.
    This is the average predicted uplift — not a causally-identified ATE
    unless identification assumptions hold.
    """
    return float(np.mean(tau_hat))

def compute_ate_error(ate_hat: float, ate_true: float) -> float:
    """
    Compute |ATE_hat - ATE_true|
    Requires ground truth ATE (IHDP/synthetic only).
    """
    return abs(ate_hat - ate_true)
