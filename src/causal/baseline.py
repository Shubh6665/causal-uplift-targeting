import numpy as np
import pandas as pd
from typing import Dict, Union

def difference_in_means(Y: Union[pd.Series, np.ndarray], T: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """
    Compute the basic difference-in-means ATE.
    Formula: ATE_hat = mean(Y | T=1) - mean(Y | T=0)
    
    Returns a dictionary with standard statistics.
    """
    Y = np.asarray(Y)
    T = np.asarray(T)
    
    treated = Y[T == 1]
    control = Y[T == 0]
    
    treated_mean = float(np.mean(treated)) if len(treated) > 0 else 0.0
    control_mean = float(np.mean(control)) if len(control) > 0 else 0.0
    
    ate_hat = treated_mean - control_mean
    
    return {
        "ate_hat": ate_hat,
        "treated_mean": treated_mean,
        "control_mean": control_mean,
        "treated_n": len(treated),
        "control_n": len(control)
    }

def difference_in_means_ci(Y: Union[pd.Series, np.ndarray], T: Union[pd.Series, np.ndarray], alpha: float = 0.05, n_bootstraps: int = 1000) -> Dict[str, float]:
    """
    Compute difference in means with a basic bootstrap CI.
    (Optional/deferred in initial phase, implemented as placeholder).
    """
    # Implemented later if bootstrap CI is requested
    pass
