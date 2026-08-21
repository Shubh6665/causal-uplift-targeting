import numpy as np
import pandas as pd
from typing import Tuple, Union
import logging

logger = logging.getLogger(__name__)

class SLearner:
    """
    Single-Model Learner (S-Learner).
    Trains a single base learner on (X, T) to predict Y.
    tau_hat(x) = mu_hat(x, T=1) - mu_hat(x, T=0)
    """
    def __init__(self, base_learner_class, base_learner_params: dict):
        self.model = base_learner_class(**base_learner_params)
        self.is_fitted = False
        
    def _prepare_features(self, X: pd.DataFrame, T: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """Helper to append treatment column to features."""
        X_copy = X.copy()
        X_copy['T_model_input'] = T
        return X_copy

    def fit(self, X: pd.DataFrame, T: Union[pd.Series, np.ndarray], Y: Union[pd.Series, np.ndarray]):
        logger.info("Fitting S-Learner...")
        X_with_t = self._prepare_features(X, T)
        self.model.fit(X_with_t, Y)
        self.is_fitted = True
        return self

    def predict_outcomes(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns predicted counterfactuals (mu_hat_1, mu_hat_0)."""
        if not self.is_fitted:
            raise ValueError("S-Learner is not fitted yet.")
            
        # Predict as if everyone is treated
        X1 = self._prepare_features(X, np.ones(len(X)))
        
        # Predict as if everyone is control
        X0 = self._prepare_features(X, np.zeros(len(X)))
        
        # If classifier, use predict_proba for class 1
        if hasattr(self.model, "predict_proba"):
            mu_hat_1 = self.model.predict_proba(X1)[:, 1]
            mu_hat_0 = self.model.predict_proba(X0)[:, 1]
        else:
            mu_hat_1 = self.model.predict(X1)
            mu_hat_0 = self.model.predict(X0)
            
        return mu_hat_1, mu_hat_0

    def predict_tau(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted treatment effects tau_hat."""
        mu_hat_1, mu_hat_0 = self.predict_outcomes(X)
        return mu_hat_1 - mu_hat_0
        
    def predict_factual(self, X: pd.DataFrame, T: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Returns predictions for the actual assigned treatment (mu_hat(x, T_observed)).
        Used for factual performance evaluation (e.g., E4 log-loss).
        """
        if not self.is_fitted:
            raise ValueError("S-Learner is not fitted yet.")
            
        X_with_t = self._prepare_features(X, T)
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_with_t)[:, 1]
        else:
            return self.model.predict(X_with_t)
