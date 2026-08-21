import argparse
import yaml
import logging
import numpy as np
from src.data.load import load_ihdp_multi
from src.data.preprocess import extract_X_T_Y, create_splits
from src.causal.s_learner import SLearner
from src.causal.t_learner import TLearner
from src.causal.x_learner import XLearner
from src.causal.propensity import get_propensity
from src.evaluation.pehe import compute_pehe
from src.evaluation.ate import mean_predicted_effect
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_e4(config_path: str):
    logger.info("=== Starting E4: Predictive vs Causal Metrics ===")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    xgb_params = {k: v for k, v in config['base_learner']['params'].items() if k != 'note'}
    
    # IHDP: continuous Y, so factual metric = MSE (not log-loss since Y is not binary)
    # Criteo: binary Y, factual metric = log-loss / PR-AUC (run when Criteo data is available)
    ihdp_data_list = load_ihdp_multi(config)
    df, tau_true, _ = ihdp_data_list[0]
    
    X, T, Y = extract_X_T_Y(df, config, 'ihdp')
    splits = create_splits(X, T, Y, config)
    
    X_train, T_train, Y_train = splits['train']
    X_test, T_test, Y_test = splits['test']
    tau_true_test = tau_true[X_test.index]
    
    s_learner = SLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
    t_learner = TLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
    
    prop_model = LogisticRegression(max_iter=1000)
    prop_model.fit(X_train, T_train)
    prop_config = {'assignment_type': 'estimated', 'fitted_model': prop_model}
    x_learner = XLearner(XGBRegressor, xgb_params).fit(
        X_train, T_train, Y_train, propensity_config=prop_config
    )
    
    # --- Factual Prediction (IHDP is continuous, so we use MSE) ---
    logger.info("--- Factual Prediction (Test Set) ---")
    logger.info("(IHDP Y is continuous; using MSE. For Criteo binary Y, use log-loss/PR-AUC.)")
    
    for name, learner in [('S-Learner', s_learner), ('T-Learner', t_learner)]:
        y_pred = learner.predict_factual(X_test, np.asarray(T_test))
        mse = float(np.mean((np.asarray(Y_test) - y_pred)**2))
        logger.info(f"  {name} | Factual MSE: {mse:.4f}")
    
    logger.info("  X-Learner: skipped (primary output is tau_hat, not factual outcome)")
        
    # --- Causal Prediction ---
    logger.info("--- Causal Prediction (Test Set) ---")
    for name, learner in [('S-Learner', s_learner), ('T-Learner', t_learner), ('X-Learner', x_learner)]:
        tau_hat = learner.predict_tau(X_test)
        pehe = compute_pehe(tau_hat, tau_true_test, dataset_name='ihdp')
        mpe = mean_predicted_effect(tau_hat)
        logger.info(f"  {name} | Causal PEHE: {pehe:.4f} | Mean Predicted Effect: {mpe:.4f}")

    logger.info("\n--- Key Insight ---")
    logger.info("If the model with better factual MSE does NOT have the best PEHE,")
    logger.info("that demonstrates: good outcome prediction ≠ good treatment-effect estimation.")
    logger.info("=== E4 Complete ===")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dev.yaml")
    args = parser.parse_args()
    run_e4(args.config)
