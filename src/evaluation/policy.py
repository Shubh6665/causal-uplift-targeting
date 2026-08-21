import numpy as np
import pandas as pd
from typing import List, Optional

def targeting_decision(tau_hat: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Return binary targeting decision: 1 if tau_hat > threshold, else 0.
    """
    return (tau_hat > threshold).astype(int)

def compute_topk_fraction_stats(Y: np.ndarray, T: np.ndarray, tau_hat: np.ndarray, k_fractions: List[float]) -> pd.DataFrame:
    """
    Computes business-friendly stats for top-k% targeted populations.
    E.g. "If we target the top 10% based on our model, what is the realized conversion rate?"
    """
    Y = np.asarray(Y)
    T = np.asarray(T)
    
    n = len(Y)
    indices = np.argsort(-tau_hat)
    
    results = []
    
    for k in k_fractions:
        top_n = int(n * k)
        if top_n == 0:
            continue
            
        idx_k = indices[:top_n]
        Y_k = Y[idx_k]
        T_k = T[idx_k]
        
        n_treated = np.sum(T_k == 1)
        n_control = np.sum(T_k == 0)
        
        conv_treated = np.sum(Y_k[T_k == 1]) / n_treated if n_treated > 0 else 0
        conv_control = np.sum(Y_k[T_k == 0]) / n_control if n_control > 0 else 0
        
        realized_uplift = conv_treated - conv_control
        
        results.append({
            "top_fraction": k,
            "targeted_users": top_n,
            "treated_users": n_treated,
            "control_users": n_control,
            "treated_conv_rate": conv_treated,
            "control_conv_rate": conv_control,
            "realized_uplift": realized_uplift
        })
        
    return pd.DataFrame(results)

def compute_ipw_policy_value(Y: np.ndarray, T: np.ndarray, tau_hat: np.ndarray, propensity: np.ndarray, threshold: float = 0.0) -> float:
    """
    Optional: IPW-based policy value estimation.
    V(pi) = E[ (I(pi(X)=T) / P(T|X)) * Y ]
    """
    pi_x = targeting_decision(tau_hat, threshold)
    
    # Clip propensity to avoid division by zero
    p_t = np.clip(propensity, 0.01, 0.99)
    # The probability of the observed treatment
    p_obs = T * p_t + (1 - T) * (1 - p_t)
    
    # Indicator if policy matched observed treatment
    matched = (pi_x == T).astype(float)
    
    value = np.mean((matched / p_obs) * Y)
    return float(value)
