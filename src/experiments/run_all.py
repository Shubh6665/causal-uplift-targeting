import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_all(config_path: str):
    logger.info(f"=== Running ALL MUST-HAVE Experiments with {config_path} ===")
    
    scripts = [
        "src.experiments.e0_baseline",
        "src.experiments.e1_estimator_comparison",
        "src.experiments.e2_imbalance",
        "src.experiments.e3_overlap",
        "src.experiments.e4_predictive_vs_causal"
    ]
    
    for script in scripts:
        cmd = ["python", "-m", script, "--config", config_path]
        logger.info(f"Executing: {' '.join(cmd)}")
        try:
            # We use check=True to stop if a script fails
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Experiment {script} failed. See logs above. Exiting run_all.")
            return
            
    logger.info("=== All MUST-HAVE Experiments Completed Successfully ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/full.yaml")
    args = parser.parse_args()
    run_all(args.config)
