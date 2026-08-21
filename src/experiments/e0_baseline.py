import argparse
import yaml
import logging
from src.data.load import load_criteo
from src.data.preprocess import extract_X_T_Y
from src.causal.baseline import difference_in_means

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_e0(config_path: str):
    logger.info("=== Starting E0: Baseline ATE (Real Criteo) ===")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    df = load_criteo(config, use_dev=True)
    
    _, T, Y = extract_X_T_Y(df, config, 'criteo')
    
    results = difference_in_means(Y, T)
    
    logger.info("=== E0 Results ===")
    logger.info(f"Treated N: {results['treated_n']}")
    logger.info(f"Control N: {results['control_n']}")
    logger.info(f"Treated Mean: {results['treated_mean']:.6f}")
    logger.info(f"Control Mean: {results['control_mean']:.6f}")
    logger.info(f"Difference in Means (ATE_hat): {results['ate_hat']:.6f}")
    
    if results['ate_hat'] < 0:
        logger.warning("ATE is negative! This indicates a potential pipeline issue or unexpected dataset property.")
    elif results['ate_hat'] > 0.5:
        logger.warning("ATE is implausibly large! Check for leakage.")
    else:
        logger.info("Baseline sanity check passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E0: Baseline Sanity Check")
    parser.add_argument("--config", type=str, default="configs/dev.yaml", help="Path to config file")
    args = parser.parse_args()
    
    run_e0(args.config)
