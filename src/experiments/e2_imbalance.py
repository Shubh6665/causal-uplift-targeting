import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from src.data.preprocess import create_synthetic_dgp
from src.causal.s_learner import SLearner
from src.causal.t_learner import TLearner
from src.causal.x_learner import XLearner
from src.evaluation.pehe import compute_pehe
from src.evaluation.ate import mean_predicted_effect, compute_ate_error
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_e2(config_path: str):
    logger.info("=== Starting E2: Treatment Imbalance (Synthetic DGP, Multi-Seed) ===")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    xgb_params = {k: v for k, v in config['base_learner']['params'].items() if k != 'note'}
    imbalance_probs = config['experiment']['imbalance_treatment_probs']
    seeds = config['experiment']['seeds']
    
    n = 50000
    n_features = 20
    
    results = {p: {'S': [], 'T': [], 'X': []} for p in imbalance_probs}
    
    for seed in seeds:
        logger.info(f"\n--- Running Seed: {seed} ---")
        
        # 1. Generate the fixed population (X) and potential outcomes
        X_baseline, _, Y_baseline, tau_true, prop_true = create_synthetic_dgp(
            n=n, n_features=n_features, alpha=1.0, treatment_prob=0.5, seed=seed
        )
        
        # Fixed test set
        X_test, T_test, Y_test, tau_true_test, _ = create_synthetic_dgp(
            n=10000, n_features=n_features, alpha=1.0, treatment_prob=0.5, seed=seed+999
        )
        ate_true = float(np.mean(tau_true_test))
        
        # 2. Iterate over different treatment probabilities
        for p in imbalance_probs:
            X_train, T_train, Y_train, _, _ = create_synthetic_dgp(
                n=n, n_features=n_features, alpha=1.0, treatment_prob=p, seed=seed+int(p*100), X_fixed=X_baseline
            )
            
            s_learner = SLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
            t_learner = TLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
            
            prop_config = {'assignment_type': 'constant', 'known_probability': p}
            x_learner = XLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train, propensity_config=prop_config)
            
            for name, learner in [('S', s_learner), ('T', t_learner), ('X', x_learner)]:
                tau_hat = learner.predict_tau(X_test)
                pehe = compute_pehe(tau_hat, tau_true_test, dataset_name='synthetic')
                results[p][name].append(pehe)
                
    logger.info("\n=== E2 Final Results (Mean ± Std over seeds) ===")
    for p in imbalance_probs:
        logger.info(f"P(T=1) = {p:.2f}:")
        for name in ['S', 'T', 'X']:
            pehes = results[p][name]
            logger.info(f"  {name}-Learner | PEHE: {np.mean(pehes):.4f} ± {np.std(pehes):.4f}")

    logger.info("=== E2 Complete ===")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/full.yaml")
    args = parser.parse_args()
    run_e2(args.config)
