import argparse
import yaml
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def audit_criteo(config_path: str):
    logger.info("=== Auditing Real Criteo Dataset ===")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    path = config['dataset']['criteo']['path']
    chunk_size = 1000000
    
    total_rows = 0
    treated_count = 0
    converted_count = 0
    treated_converted = 0
    control_converted = 0
    visited_count = 0
    exposed_count = 0
    exposed_control = 0
    
    feature_sums = None
    feature_sq_sums = None
    columns = None
    
    logger.info(f"Streaming dataset from {path} in chunks of {chunk_size}...")
    
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
        if i == 0:
            columns = list(chunk.columns)
            feature_cols = [c for c in columns if c.startswith('f')]
            feature_sums = np.zeros(len(feature_cols))
            feature_sq_sums = np.zeros(len(feature_cols))
            
        total_rows += len(chunk)
        
        # Treatment
        t_mask = chunk['treatment'] == 1
        treated_count += t_mask.sum()
        
        # Outcomes
        converted_count += chunk['conversion'].sum()
        visited_count += chunk['visit'].sum()
        exposed_count += chunk['exposure'].sum()
        
        treated_converted += chunk.loc[t_mask, 'conversion'].sum()
        control_converted += chunk.loc[~t_mask, 'conversion'].sum()
        exposed_control += chunk.loc[~t_mask, 'exposure'].sum()
        
        # Feature stats
        f_data = chunk[feature_cols].values
        feature_sums += f_data.sum(axis=0)
        feature_sq_sums += (f_data ** 2).sum(axis=0)
        
        logger.info(f"Processed chunk {i+1} ({total_rows} rows total so far)")
        
    # Final computation
    control_count = total_rows - treated_count
    
    treated_conv_rate = treated_converted / treated_count
    control_conv_rate = control_converted / control_count
    ate_baseline = treated_conv_rate - control_conv_rate
    
    feature_means = feature_sums / total_rows
    feature_vars = (feature_sq_sums / total_rows) - (feature_means ** 2)
    feature_stds = np.sqrt(feature_vars)
    
    report = f"""# Criteo Data Audit (Real Dataset)

## 1. File Metadata
- **Source**: Criteo AI Lab Official Unbiased Release (v2.1)
- **Local Path**: `{path}`
- **Total Rows**: {total_rows:,}
- **Total Columns**: {len(columns)}

## 2. Schema
- **Features**: {len(feature_cols)} columns (`f0` to `f11`)
- **Treatment**: `treatment` (binary)
- **Outcomes**: `conversion` (binary), `visit` (binary)
- **Leakage variable**: `exposure` (binary)

## 3. Global Statistics
- **Treatment Count**: {treated_count:,}
- **Control Count**: {control_count:,}
- **Treatment Ratio**: {(treated_count / total_rows) * 100:.2f}%
- **Total Conversions**: {converted_count:,} ({(converted_count / total_rows) * 100:.4f}%)
- **Total Visits**: {visited_count:,} ({(visited_count / total_rows) * 100:.2f}%)
- **Total Exposures**: {exposed_count:,} ({(exposed_count / total_rows) * 100:.2f}%)

## 4. Integrity Checks
- **Exposure in Control Group**: {exposed_control} (MUST be 0. Proves exposure requires treatment).
- **Leakage check**: `exposure` must NEVER be used as a feature, as it is post-treatment.

## 5. Baseline Difference in Means (ATE)
- **Treated Conversion Rate**: {treated_conv_rate:.6f}
- **Control Conversion Rate**: {control_conv_rate:.6f}
- **Difference (Uplift)**: +{ate_baseline:.6f}

## 6. Feature Statistics
| Feature | Mean | Std Dev |
|---------|------|---------|
"""
    for i, col in enumerate(feature_cols):
        report += f"| {col} | {feature_means[i]:.4f} | {feature_stds[i]:.4f} |\n"
        
    with open('reports/criteo_data_audit.md', 'w') as f:
        f.write(report)
        
    logger.info("Audit complete! Saved to reports/criteo_data_audit.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/full.yaml")
    args = parser.parse_args()
    audit_criteo(args.config)
