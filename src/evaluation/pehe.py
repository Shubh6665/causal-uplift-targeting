import numpy as np
from typing import List, Optional

def compute_pehe(tau_hat: np.ndarray, tau_true: np.ndarray, dataset_name: Optional[str] = None) -> float:
    """
    Compute Precision in Estimation of Heterogeneous Effect (PEHE).
    Formula: PEHE = sqrt( (1/n) * sum_i (tau_hat_i - tau_i)^2 )
    
    IHDP/synthetic ONLY.
    Raises ValueError if dataset_name == 'criteo'.
    """
    if dataset_name and dataset_name.lower() == 'criteo':
        raise ValueError(
            "PEHE requires individual ground-truth effects. "
            "Criteo does not expose both counterfactuals. "
            "Use Qini/AUUC for Criteo evaluation."
        )
        
    return float(np.sqrt(np.mean((tau_hat - tau_true)**2)))

def compute_pehe_multi(tau_hat_list: List[np.ndarray], tau_true_list: List[np.ndarray]) -> dict:
    """
    Run across multiple IHDP replications.
    Returns {mean_pehe, std_pehe, all_pehe_values}
    """
    if len(tau_hat_list) != len(tau_true_list):
        raise ValueError("List lengths must match.")
        
    pehe_values = []
    for th, tt in zip(tau_hat_list, tau_true_list):
        pehe_values.append(compute_pehe(th, tt))
        
    return {
        "mean_pehe": float(np.mean(pehe_values)),
        "std_pehe": float(np.std(pehe_values)),
        "all_pehe_values": pehe_values
    }
