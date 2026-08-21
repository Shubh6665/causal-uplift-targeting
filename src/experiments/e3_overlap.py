import argparse
import yaml
import logging
import numpy as np
from src.data.preprocess import create_synthetic_dgp
from src.causal.s_learner import SLearner
from src.causal.t_learner import TLearner
from src.causal.x_learner import XLearner
from src.evaluation.pehe import compute_pehe
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_e3(config_path: str):
    logger.info("=== Starting E3: Overlap Stress Test (Synthetic DGP, Multi-Seed) ===")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    xgb_params = {k: v for k, v in config['base_learner']['params'].items() if k != 'note'}
    alpha_values = config['experiment']['overlap_alpha_values']
    seeds = config['experiment']['seeds']
    
    n_train = 50000
    n_test = 10000
    n_features = 20
    
    results = {alpha: {'S': [], 'T': [], 'X': []} for alpha in alpha_values}
    
    for seed in seeds:
        logger.info(f"\n--- Running Seed: {seed} ---")
        
        for alpha in alpha_values:
            X_train, T_train, Y_train, _, prop_true_train = create_synthetic_dgp(
                n=n_train, n_features=n_features, alpha=alpha, treatment_prob=0.5, seed=seed
            )
            
            X_test, T_test, Y_test, tau_true_test, prop_true_test = create_synthetic_dgp(
                n=n_test, n_features=n_features, alpha=alpha, treatment_prob=0.5, seed=seed+999
            )
            
            s_learner = SLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
            t_learner = TLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
            
            prop_config = {'assignment_type': 'true_known', 'true_propensities': prop_true_test}
            x_learner = XLearner(XGBRegressor, xgb_params).fit(
                X_train, T_train, Y_train, propensity_config=prop_config
            )
            
            for name, learner in [('S', s_learner), ('T', t_learner), ('X', x_learner)]:
                tau_hat = learner.predict_tau(X_test)
                pehe = compute_pehe(tau_hat, tau_true_test, dataset_name='synthetic')
                results[alpha][name].append(pehe)

    logger.info("\n=== E3 Final Results (Mean ± Std over seeds) ===")
    for alpha in alpha_values:
        logger.info(f"Alpha = {alpha}:")
        for name in ['S', 'T', 'X']:
            pehes = results[alpha][name]
            logger.info(f"  {name}-Learner | PEHE: {np.mean(pehes):.4f} ± {np.std(pehes):.4f}")

    logger.info("=== E3 Complete ===")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/full.yaml")
    args = parser.parse_args()
    run_e3(args.config)
