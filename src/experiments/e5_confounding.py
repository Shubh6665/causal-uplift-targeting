import argparse
import yaml
import logging
import numpy as np
from src.data.preprocess import create_synthetic_dgp
from src.causal.s_learner import SLearner
from src.causal.t_learner import TLearner
from src.causal.x_learner import XLearner
from src.evaluation.pehe import compute_pehe
from src.evaluation.ate import mean_predicted_effect, compute_ate_error
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_e5(config_path: str):
    logger.info("=== Starting E5: Hidden Confounding (Synthetic DGP) ===")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    xgb_params = {k: v for k, v in config['base_learner']['params'].items() if k != 'note'}
    seed = config['experiment']['seeds'][0]
    
    n_train = 20000
    n_test = 5000
    n_features = 20
    gamma_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    for gamma in gamma_values:
        logger.info(f"\n--- Confounder Strength: gamma = {gamma} ---")
        
        X_train, T_train, Y_train, tau_true_train, prop_true_train = create_synthetic_dgp(
            n=n_train, n_features=n_features, alpha=1.0, treatment_prob=0.5,
            seed=seed, add_hidden_confounder=(gamma > 0), confounder_strength=gamma
        )
        
        X_test, T_test, Y_test, tau_true_test, prop_true_test = create_synthetic_dgp(
            n=n_test, n_features=n_features, alpha=1.0, treatment_prob=0.5,
            seed=seed+999, add_hidden_confounder=(gamma > 0), confounder_strength=gamma
        )
        
        ate_true = float(np.mean(tau_true_test))
        
        # Estimator only sees X, not U — this is the key
        s_learner = SLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
        t_learner = TLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
        
        prop_config = {'assignment_type': 'true_known', 'true_propensities': prop_true_test}
        x_learner = XLearner(XGBRegressor, xgb_params).fit(
            X_train, T_train, Y_train, propensity_config=prop_config
        )
        
        for name, learner in [('S-Learner', s_learner), ('T-Learner', t_learner), ('X-Learner', x_learner)]:
            tau_hat = learner.predict_tau(X_test)
            mpe = mean_predicted_effect(tau_hat)
            abs_ate_err = abs(mpe - ate_true)
            pehe = compute_pehe(tau_hat, tau_true_test, dataset_name='synthetic')
            logger.info(f"  {name} | PEHE: {pehe:.4f} | Absolute ATE Error: {abs_ate_err:.4f} | Mean Pred Effect: {mpe:.4f}")
    
    logger.info("\n--- Interpretation ---")
    logger.info("As gamma increases, hidden confounding produces increasing absolute ATE estimation error.")
    logger.info("This demonstrates sensitivity to violated unconfoundedness;")
    logger.info("it does NOT demonstrate that our models solve hidden confounding.")
    logger.info("Criteo mitigates this via its randomized experimental design.")
    logger.info("=== E5 Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dev.yaml")
    args = parser.parse_args()
    run_e5(args.config)
