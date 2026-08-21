import numpy as np
import pandas as pd
from typing import Tuple, Union
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

class TLearner:
    """
    Two-Model Learner (T-Learner).
    Trains two separate base learners: one on T=1 data, one on T=0 data.
    tau_hat(x) = mu_hat_1(x) - mu_hat_0(x)
    """
    def __init__(self, base_learner_class, base_learner_params: dict):
        self.model_1 = base_learner_class(**base_learner_params)
        self.model_0 = base_learner_class(**base_learner_params)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, T: Union[pd.Series, np.ndarray], Y: Union[pd.Series, np.ndarray]):
        logger.info("Fitting T-Learner...")
        T = np.asarray(T)
        Y = np.asarray(Y)
        
        # Split data by treatment arm
        treated_mask = (T == 1)
        control_mask = (T == 0)
        
        X1, Y1 = X[treated_mask], Y[treated_mask]
        X0, Y0 = X[control_mask], Y[control_mask]
        
        if len(X1) == 0 or len(X0) == 0:
            raise ValueError("Data must contain both treated and control units.")
            
        # Fit models
        self.model_1.fit(X1, Y1)
        self.model_0.fit(X0, Y0)
        
        self.is_fitted = True
        return self

    def predict_outcomes(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns predicted counterfactuals (mu_hat_1, mu_hat_0)."""
        if not self.is_fitted:
            raise ValueError("T-Learner is not fitted yet.")
            
        if hasattr(self.model_1, "predict_proba"):
            mu_hat_1 = self.model_1.predict_proba(X)[:, 1]
            mu_hat_0 = self.model_0.predict_proba(X)[:, 1]
        else:
            mu_hat_1 = self.model_1.predict(X)
            mu_hat_0 = self.model_0.predict(X)
            
        return mu_hat_1, mu_hat_0

    def predict_tau(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted treatment effects tau_hat."""
        mu_hat_1, mu_hat_0 = self.predict_outcomes(X)
        return mu_hat_1 - mu_hat_0
        
    def predict_factual(self, X: pd.DataFrame, T: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Returns predictions for the actual assigned treatment.
        Used for factual performance evaluation (e.g., E4 log-loss).
        """
        if not self.is_fitted:
            raise ValueError("T-Learner is not fitted yet.")
            
        T = np.asarray(T)
        mu_hat_1, mu_hat_0 = self.predict_outcomes(X)
        
        # Combine based on observed T
        # mu_hat_factual = mu1 if T==1 else mu0
        return (T * mu_hat_1) + ((1 - T) * mu_hat_0)
