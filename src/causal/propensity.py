import numpy as np
import pandas as pd
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

def get_propensity(
    X: pd.DataFrame, 
    T: Union[pd.Series, np.ndarray], 
    assignment_type: str, 
    known_probability: Optional[float] = None,
    learner_class=None, 
    learner_params: Optional[dict] = None,
    true_propensities: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Get propensity scores based on the experiment design (v2.1 fix).
    
    assignment_type:
      'constant'   -> returns np.full(len(X), known_probability)
                      Use for Criteo (~0.85); interview: "randomized, known"
      'estimated'  -> fits learner_class on (X, T); returns predict_proba
                      Use for IHDP (observational-style)
      'true_known' -> returns true_propensities array
                      Use for synthetic DGP (exact ground truth)
    """
    n = len(X)
    
    if assignment_type == 'constant':
        if known_probability is None:
            raise ValueError("known_probability must be provided for 'constant' assignment_type.")
        logger.info(f"Propensity: Using known constant probability = {known_probability}")
        return np.full(n, known_probability)
        
    elif assignment_type == 'true_known':
        if true_propensities is None or len(true_propensities) != n:
            raise ValueError("true_propensities array matching X length must be provided for 'true_known'.")
        logger.info("Propensity: Using true known propensities from DGP.")
        return np.asarray(true_propensities)
        
    elif assignment_type == 'estimated':
        if learner_class is None:
            from sklearn.linear_model import LogisticRegression
            logger.info("Propensity: No learner provided, defaulting to LogisticRegression.")
            learner_class = LogisticRegression
            learner_params = learner_params or {'max_iter': 1000}
            
        logger.info(f"Propensity: Estimating using {learner_class.__name__}.")
        model = learner_class(**(learner_params or {}))
        model.fit(X, T)
        
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        else:
            # For linear models like RidgeClassifier
            decision = model.decision_function(X)
            # pseudo-probability via sigmoid
            return 1 / (1 + np.exp(-decision))
            
    else:
        raise ValueError(f"Unknown assignment_type: {assignment_type}")

def check_overlap(propensity_scores: np.ndarray, T: np.ndarray, threshold: float = 0.05) -> dict:
    """Check what percentage of units have propensity near 0 or 1."""
    near_0 = np.mean(propensity_scores < threshold)
    near_1 = np.mean(propensity_scores > (1 - threshold))
    
    logger.info(f"Overlap check: {near_0*100:.2f}% < {threshold}, {near_1*100:.2f}% > {1-threshold}")
    
    return {
        "near_0_pct": near_0,
        "near_1_pct": near_1,
        "total_extreme_pct": near_0 + near_1
    }

def clip_propensity(scores: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """Clip propensities to avoid division by zero in IPW."""
    return np.clip(scores, lower, upper)
