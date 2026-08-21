import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from src.data.load import load_criteo
from src.data.preprocess import extract_X_T_Y, create_splits
from src.causal.s_learner import SLearner
from src.causal.t_learner import TLearner
from src.causal.x_learner import XLearner
from src.evaluation.uplift import compute_qini_coefficient, compute_auuc
from xgboost import XGBRegressor
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_e1_criteo(config_path: str):
    logger.info("=== Starting E1: S/T/X on Real Criteo ===")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    xgb_params = {k: v for k, v in config['base_learner']['params'].items() if k != 'note'}
    
    df = load_criteo(config, use_dev=False)
    
    X, T, Y = extract_X_T_Y(df, config, 'criteo')
    splits = create_splits(X, T, Y, config)
    
    X_train, T_train, Y_train = splits['train']
    X_test, T_test, Y_test = splits['test']
    
    # Criteo uses constant known propensity (~0.85)
    prop_config = {'assignment_type': 'constant', 'known_probability': 0.85}
    
    logger.info("Fitting S-Learner...")
    s_learner = SLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
    logger.info("Fitting T-Learner...")
    t_learner = TLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
    logger.info("Fitting X-Learner...")
    x_learner = XLearner(XGBRegressor, xgb_params).fit(
        X_train, T_train, Y_train, propensity_config=prop_config
    )
    
    results = []
    
    for name, learner in [('S-Learner', s_learner), ('T-Learner', t_learner), ('X-Learner', x_learner)]:
        tau_hat = learner.predict_tau(X_test)
        
        qini_coeff = compute_qini_coefficient(np.asarray(Y_test), np.asarray(T_test), tau_hat)
        auuc_val = compute_auuc(np.asarray(Y_test), np.asarray(T_test), tau_hat)
        
        results.append({
            'Estimator': name,
            'Qini_Coefficient': qini_coeff,
            'AUUC': auuc_val
        })
        
        logger.info(f"{name} | Qini Coeff: {qini_coeff:.6f} | AUUC: {auuc_val:.6f}")
        
    os.makedirs('reports/results', exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv('reports/results/criteo_e1_results.csv', index=False)
    logger.info("Saved results to reports/results/criteo_e1_results.csv")
    logger.info(f"Evaluation used {len(X_test)} units.")
    
    logger.info("=== E1 Criteo Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/full.yaml")
    args = parser.parse_args()
    run_e1_criteo(args.config)
