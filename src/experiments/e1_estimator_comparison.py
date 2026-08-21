import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from src.data.load import load_ihdp_multi
from src.data.preprocess import extract_X_T_Y, create_splits
from src.causal.s_learner import SLearner
from src.causal.t_learner import TLearner
from src.causal.x_learner import XLearner
from src.causal.propensity import get_propensity
from src.evaluation.pehe import compute_pehe, compute_pehe_multi
from src.evaluation.ate import mean_predicted_effect, compute_ate_error
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_e1(config_path: str):
    logger.info("=== Starting E1: Estimator Comparison (IHDP) ===")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    xgb_params = {k: v for k, v in config['base_learner']['params'].items() if k != 'note'}
    
    ihdp_data_list = load_ihdp_multi(config)
    
    all_pehes = {'S-Learner': [], 'T-Learner': [], 'X-Learner': []}
    all_ate_errors = {'S-Learner': [], 'T-Learner': [], 'X-Learner': []}
    
    for idx, (df, tau_true, _) in enumerate(ihdp_data_list):
        logger.info(f"--- Running IHDP Replication {idx+1} ---")
        X, T, Y = extract_X_T_Y(df, config, 'ihdp')
        
        splits = create_splits(X, T, Y, config)
        X_train, T_train, Y_train = splits['train']
        X_test, T_test, Y_test = splits['test']
        
        test_indices = X_test.index
        tau_true_test = tau_true[test_indices]
        ate_true = float(np.mean(tau_true_test))
        
        s_learner = SLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
        t_learner = TLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
        
        # IHDP uses estimated propensity — fit it during training, pass fitted model
        prop_model = LogisticRegression(max_iter=1000)
        prop_model.fit(X_train, T_train)
        prop_config = {
            'assignment_type': 'estimated',
            'fitted_model': prop_model
        }
        x_learner = XLearner(XGBRegressor, xgb_params).fit(
            X_train, T_train, Y_train, propensity_config=prop_config
        )
        
        for name, learner in [('S-Learner', s_learner), ('T-Learner', t_learner), ('X-Learner', x_learner)]:
            tau_hat = learner.predict_tau(X_test)
            pehe = compute_pehe(tau_hat, tau_true_test)
            ate_err = compute_ate_error(mean_predicted_effect(tau_hat), ate_true)
            
            all_pehes[name].append(pehe)
            all_ate_errors[name].append(ate_err)
            logger.info(f"  {name} | PEHE: {pehe:.4f} | ATE Error: {ate_err:.4f}")
    
    logger.info("\n=== E1 Summary (across replications) ===")
    for name in all_pehes:
        pehes = all_pehes[name]
        ate_errs = all_ate_errors[name]
        logger.info(
            f"{name} | PEHE: {np.mean(pehes):.4f} ± {np.std(pehes):.4f} | "
            f"ATE Error: {np.mean(ate_errs):.4f} ± {np.std(ate_errs):.4f}"
        )
    
    logger.info("=== E1 Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dev.yaml")
    args = parser.parse_args()
    run_e1(args.config)
