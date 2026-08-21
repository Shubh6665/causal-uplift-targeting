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
from src.evaluation.policy import compute_topk_fraction_stats
from xgboost import XGBRegressor
from causalml.metrics import auuc_score, qini_score

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def bootstrap_topk(Y, T, tau_hat, k_fractions, n_bootstrap=100):
    np.random.seed(42)
    n = len(Y)
    results = {k: [] for k in k_fractions}
    results['ate'] = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        Y_boot = Y[indices]
        T_boot = T[indices]
        tau_boot = tau_hat[indices]
        
        # ATE
        treated_conv = Y_boot[T_boot == 1].mean() if (T_boot == 1).sum() > 0 else 0
        control_conv = Y_boot[T_boot == 0].mean() if (T_boot == 0).sum() > 0 else 0
        results['ate'].append(treated_conv - control_conv)
        
        # Top K
        stats = compute_topk_fraction_stats(Y_boot, T_boot, tau_boot, k_fractions)
        for _, row in stats.iterrows():
            frac = row['top_fraction']
            results[frac].append(row['realized_uplift'])
            
    # Calculate CIs
    cis = {}
    for k in k_fractions:
        cis[k] = {
            'mean': np.mean(results[k]),
            'lower': np.percentile(results[k], 2.5),
            'upper': np.percentile(results[k], 97.5)
        }
    cis['ate'] = {
        'mean': np.mean(results['ate']),
        'lower': np.percentile(results['ate'], 2.5),
        'upper': np.percentile(results['ate'], 97.5)
    }
    return cis

def run_audit(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    xgb_params = {k: v for k, v in config['base_learner']['params'].items() if k != 'note'}
    df = load_criteo(config, use_dev=False) # 5%
    X, T, Y = extract_X_T_Y(df, config, 'criteo')
    splits = create_splits(X, T, Y, config)
    
    X_train, T_train, Y_train = splits['train']
    X_test, T_test, Y_test = splits['test']
    
    prop_config = {'assignment_type': 'constant', 'known_probability': 0.85}
    
    # Train
    s_learner = SLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
    t_learner = TLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train)
    x_learner = XLearner(XGBRegressor, xgb_params).fit(X_train, T_train, Y_train, propensity_config=prop_config)
    
    # Task 2: Cross check
    results = []
    
    for name, learner in [('S-Learner', s_learner), ('T-Learner', t_learner), ('X-Learner', x_learner)]:
        tau_hat = learner.predict_tau(X_test)
        
        our_qini = compute_qini_coefficient(np.asarray(Y_test), np.asarray(T_test), tau_hat)
        our_auuc = compute_auuc(np.asarray(Y_test), np.asarray(T_test), tau_hat)
        
        df_pred = pd.DataFrame({
            'y': np.asarray(Y_test),
            'w': np.asarray(T_test),
            'pred': np.asarray(tau_hat)
        })
        
        cml_auuc = auuc_score(df_pred, outcome_col='y', treatment_col='w', normalize=True)['pred']
        cml_qini = qini_score(df_pred, outcome_col='y', treatment_col='w', normalize=True)['pred']
        
        results.append({
            'Model': name,
            'Our AUUC': our_auuc,
            'CausalML AUUC': cml_auuc,
            'Our Qini': our_qini,
            'CausalML Qini': cml_qini
        })
        
        if name == 'S-Learner':
            # Task 4: Bootstrap Top-K CIs
            k_fractions = [0.01, 0.05, 0.10, 0.20, 0.50]
            cis = bootstrap_topk(np.asarray(Y_test), np.asarray(T_test), tau_hat, k_fractions, n_bootstrap=100)
            
    print("=== TASK 2: METRICS CROSS CHECK ===")
    print(pd.DataFrame(results).to_string(index=False))
    
    print("\n=== TASK 4: S-LEARNER TOP-K UNCERTAINTY (BOOTSTRAP 95% CI) ===")
    print(f"Global Test ATE: {cis['ate']['mean']:.6f} [{cis['ate']['lower']:.6f}, {cis['ate']['upper']:.6f}]")
    for k in k_fractions:
        print(f"Top {int(k*100)}%: {cis[k]['mean']:.6f} [{cis[k]['lower']:.6f}, {cis[k]['upper']:.6f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/full.yaml")
    args = parser.parse_args()
    run_audit(args.config)
