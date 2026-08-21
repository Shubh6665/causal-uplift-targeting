import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional
import logging
from src.causal.propensity import get_propensity

logger = logging.getLogger(__name__)

class XLearner:
    """
    Cross-Learner (X-Learner).
    Follows a 5-step pipeline:
    1. Fit outcome models mu0 on control, mu1 on treated.
    2. Compute pseudo-effects: D1 = Y1 - mu0(X1), D0 = mu1(X0) - Y0.
    3. Fit effect models tau1 on (X1, D1) and tau0 on (X0, D0).
    4. Compute/Get propensity scores e(x).
    5. Combine: tau(x) = (1 - e(x))*tau1(x) + e(x)*tau0(x).
    """
    def __init__(
        self, 
        base_learner_class, 
        base_learner_params: dict,
        effect_learner_class=None,
        effect_learner_params: Optional[dict] = None
    ):
        # Stage 1 models (outcomes)
        self.mu0_model = base_learner_class(**base_learner_params)
        self.mu1_model = base_learner_class(**base_learner_params)
        
        # Stage 2 models (effects)
        effect_cls = effect_learner_class or base_learner_class
        effect_params = effect_learner_params or base_learner_params
        
        self.tau0_model = effect_cls(**effect_params)
        self.tau1_model = effect_cls(**effect_params)
        
        self.is_fitted = False
        self.propensity_config = {}
        
        # Storage for debugging/inspection
        self._D0 = None
        self._D1 = None

    def fit(self, X: pd.DataFrame, T: Union[pd.Series, np.ndarray], Y: Union[pd.Series, np.ndarray], propensity_config: dict):
        logger.info("Fitting X-Learner...")
        T = np.asarray(T)
        Y = np.asarray(Y)
        
        # Split data by treatment arm
        treated_mask = (T == 1)
        control_mask = (T == 0)
        
        X1, Y1 = X[treated_mask], Y[treated_mask]
        X0, Y0 = X[control_mask], Y[control_mask]
        
        if len(X1) == 0 or len(X0) == 0:
            raise ValueError("Data must contain both treated and control units.")
            
        # Step 1: Fit outcome models
        self.mu1_model.fit(X1, Y1)
        self.mu0_model.fit(X0, Y0)
        
        # Step 2: Compute pseudo-effects
        # mu0_hat on treated
        if hasattr(self.mu0_model, "predict_proba"):
            mu0_pred_X1 = self.mu0_model.predict_proba(X1)[:, 1]
            mu1_pred_X0 = self.mu1_model.predict_proba(X0)[:, 1]
        else:
            mu0_pred_X1 = self.mu0_model.predict(X1)
            mu1_pred_X0 = self.mu1_model.predict(X0)
            
        D1 = Y1 - mu0_pred_X1
        D0 = mu1_pred_X0 - Y0
        
        self._D1, self._D0 = D1, D0
        
        # Step 3: Fit effect models
        # Note: D1 and D0 are continuous even if Y is binary, so effect models must be regressors.
        # Ensure effect_cls is a regressor (like XGBRegressor, not XGBClassifier).
        self.tau1_model.fit(X1, D1)
        self.tau0_model.fit(X0, D0)
        
        # Store propensity configuration for predict stage
        self.propensity_config = propensity_config
        self.is_fitted = True
        return self

    def predict_outcomes(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns Stage 1 predicted counterfactual outcomes (mu_hat_1, mu_hat_0)."""
        if not self.is_fitted:
            raise ValueError("X-Learner is not fitted yet.")
            
        if hasattr(self.mu1_model, "predict_proba"):
            mu_hat_1 = self.mu1_model.predict_proba(X)[:, 1]
            mu_hat_0 = self.mu0_model.predict_proba(X)[:, 1]
        else:
            mu_hat_1 = self.mu1_model.predict(X)
            mu_hat_0 = self.mu0_model.predict(X)
            
        return mu_hat_1, mu_hat_0

    def get_pseudo_effects(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return D1, D0 for inspection and debugging."""
        return self._D1, self._D0

    def get_propensity(self, X: pd.DataFrame) -> np.ndarray:
        """Helper to get propensity using the stored config."""
        # Note: T is not required here for 'constant' and 'true_known' (if passed inside config). 
        # For 'estimated', we usually need to train it during fit().
        # Let's handle estimating it properly by expecting the pre-trained or configured propensity method.
        # As per v2.1, for Criteo it's constant.
        
        # Because we only need the predictions of e(x) on X during predict_tau, 
        # if it's 'estimated', we assume a model was passed or we just evaluate constant/true_known.
        assignment_type = self.propensity_config.get('assignment_type', 'constant')
        
        if assignment_type == 'constant':
            return get_propensity(X, None, 'constant', known_probability=self.propensity_config['known_probability'])
        elif assignment_type == 'true_known':
            return get_propensity(X, None, 'true_known', true_propensities=self.propensity_config['true_propensities'])
        elif assignment_type == 'estimated':
            model = self.propensity_config.get('fitted_model')
            if model is None:
                raise ValueError("For 'estimated' propensity in X-Learner prediction, 'fitted_model' must be in propensity_config.")
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)[:, 1]
            return model.predict(X)
        else:
            raise ValueError(f"Unknown assignment_type in X-Learner: {assignment_type}")

    def predict_tau(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted treatment effects tau_hat."""
        if not self.is_fitted:
            raise ValueError("X-Learner is not fitted yet.")
            
        # Predict tau1 and tau0
        tau_hat_1 = self.tau1_model.predict(X)
        tau_hat_0 = self.tau0_model.predict(X)
        
        # Get e(x)
        e_hat = self.get_propensity(X)
        
        # Combine
        tau_hat = (1 - e_hat) * tau_hat_1 + e_hat * tau_hat_0
        return tau_hat
