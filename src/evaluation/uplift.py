import numpy as np
import pandas as pd
from typing import Tuple

from causalml.metrics import auuc_score, qini_score

def compute_qini_curve(Y: np.ndarray, T: np.ndarray, tau_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Qini curve based on the project's explicit convention:
    1. Rank units by tau_hat descending
    2. At each fraction f:
       treated_in_top = count(T=1 in top f fraction)
       control_in_top = count(T=0 in top f fraction)
       treated_total  = count(T=1 in full test)
       control_total  = count(T=0 in full test)
       qini(f) = (Y[T==1, top_f].sum() / treated_total)
                 - (Y[T==0, top_f].sum() / control_total)
                   * (treated_in_top / treated_total)
    
    Returns: (fractions, qini_values)
    """
    Y = np.asarray(Y)
    T = np.asarray(T)
    tau_hat = np.asarray(tau_hat)
    
    n = len(Y)
    treated_total = np.sum(T == 1)
    control_total = np.sum(T == 0)
    
    if treated_total == 0 or control_total == 0:
        raise ValueError("Data must have both treatment and control units.")
        
    # Sort descending by tau_hat
    indices = np.argsort(-tau_hat)
    Y_sorted = Y[indices]
    T_sorted = T[indices]
    
    # Cumulative sums
    cum_Y_treated = np.cumsum(Y_sorted * (T_sorted == 1))
    cum_Y_control = np.cumsum(Y_sorted * (T_sorted == 0))
    cum_T_treated = np.cumsum(T_sorted == 1)
    
    # Qini value calculation at each point
    qini_values = np.zeros(n + 1)
    fractions = np.linspace(0, 1, n + 1)
    
    q_vals = (cum_Y_treated / treated_total) - (cum_Y_control / control_total) * (cum_T_treated / treated_total)
    qini_values[1:] = q_vals
    
    return fractions, qini_values

def compute_qini_coefficient(Y: np.ndarray, T: np.ndarray, tau_hat: np.ndarray) -> float:
    """Compute normalized Qini coefficient using CausalML."""
    # Add tiny noise to break ties, otherwise CausalML qcut crashes if predictions are identical
    tau_hat = np.asarray(tau_hat, dtype=float)
    tau_hat += np.random.normal(0, 1e-10, size=len(tau_hat))
    
    df = pd.DataFrame({
        'y': np.asarray(Y),
        'w': np.asarray(T),
        'pred': tau_hat
    })
    return qini_score(df, outcome_col='y', treatment_col='w', normalize=True)['pred']

def compute_auuc(Y: np.ndarray, T: np.ndarray, tau_hat: np.ndarray) -> float:
    """Compute normalized AUUC using CausalML."""
    tau_hat = np.asarray(tau_hat, dtype=float)
    tau_hat += np.random.normal(0, 1e-10, size=len(tau_hat))
    
    df = pd.DataFrame({
        'y': np.asarray(Y),
        'w': np.asarray(T),
        'pred': tau_hat
    })
    return auuc_score(df, outcome_col='y', treatment_col='w', normalize=True)['pred']
