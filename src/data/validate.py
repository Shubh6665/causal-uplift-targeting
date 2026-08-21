import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ValidationReport:
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []

    def add_error(self, msg: str):
        self.is_valid = False
        self.errors.append(msg)
        logger.error(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(msg)

def validate_schema(df: pd.DataFrame, config: dict, report: ValidationReport):
    """Check for 12 features f0-f11 and ensure they are numeric."""
    features = config['dataset']['criteo']['feature_cols']
    if len(features) != 12:
        report.add_error(f"Expected 12 features, found {len(features)} in config.")
    
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        report.add_error(f"Missing features in dataframe: {missing_cols}")
        
    for col in features:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            report.add_error(f"Feature {col} is not numeric.")

def validate_treatment(df: pd.DataFrame, col: str, report: ValidationReport):
    """Ensure treatment is strictly binary {0, 1}."""
    if col not in df.columns:
        report.add_error(f"Treatment column '{col}' missing.")
        return
        
    unique_vals = set(df[col].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        report.add_error(f"Treatment column '{col}' contains non-binary values: {unique_vals}")

def validate_outcome(df: pd.DataFrame, col: str, report: ValidationReport):
    """Ensure outcome is binary."""
    if col not in df.columns:
        report.add_error(f"Outcome column '{col}' missing.")
        return
        
    unique_vals = set(df[col].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        report.add_error(f"Outcome column '{col}' contains non-binary values: {unique_vals}")

def validate_no_leakage(df: pd.DataFrame, feature_cols: List[str], report: ValidationReport):
    """Ensure exposure or other post-treatment variables are NOT in feature_cols."""
    leakage_risks = ['exposure', 'visit', 'conversion']
    for risk in leakage_risks:
        if risk in feature_cols:
            report.add_error(f"LEAKAGE RISK: '{risk}' found in feature_cols.")

def check_treatment_balance(df: pd.DataFrame, col: str, report: ValidationReport) -> dict:
    """Log the treatment ratio. Criteo should be ~0.85."""
    if col not in df.columns:
        return {}
    
    treat_ratio = df[col].mean()
    logger.info(f"Treatment ratio: {treat_ratio:.4f} (Expected ~0.85 for Criteo)")
    
    if treat_ratio < 0.1 or treat_ratio > 0.95:
        report.add_warning(f"Extreme treatment imbalance detected: {treat_ratio:.4f}")
        
    return {"treatment_ratio": treat_ratio}

def check_outcome_imbalance(df: pd.DataFrame, col: str, report: ValidationReport) -> dict:
    """Log the outcome rate. Criteo should be ~0.29%."""
    if col not in df.columns:
        return {}
        
    conversion_rate = df[col].mean()
    logger.info(f"Overall conversion rate: {conversion_rate:.6f} (Expected ~0.0029 for Criteo)")
    
    if conversion_rate < 0.01:
        report.add_warning(
            f"Severe outcome imbalance ({conversion_rate:.6f}). "
            "Classification accuracy metrics (like plain Accuracy) will be useless."
        )
        
    return {"conversion_rate": conversion_rate}

def check_missing_values(df: pd.DataFrame, report: ValidationReport) -> dict:
    """Check for nulls."""
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    
    if not cols_with_nulls.empty:
        report.add_warning(f"Missing values found: \n{cols_with_nulls}")
        
    return cols_with_nulls.to_dict()

def run_all_validations(df: pd.DataFrame, config: dict, dataset_type: str = 'criteo') -> ValidationReport:
    """Run all validation checks for the dataset."""
    report = ValidationReport()
    logger.info(f"Running validations for {dataset_type} dataset...")
    
    if dataset_type == 'criteo':
        c_config = config['dataset']['criteo']
        t_col = c_config['treatment_col']
        y_col = c_config['outcome_col']
        f_cols = c_config['feature_cols']
        
        validate_schema(df, config, report)
        validate_treatment(df, t_col, report)
        validate_outcome(df, y_col, report)
        validate_no_leakage(df, f_cols, report)
        
        check_treatment_balance(df, t_col, report)
        check_outcome_imbalance(df, y_col, report)
        check_missing_values(df, report)
        
    elif dataset_type == 'ihdp':
        # Basic validation for IHDP
        validate_treatment(df, 'treatment', report)
        validate_outcome(df, 'y_factual', report)
        check_missing_values(df, report)
        
    if not report.is_valid:
        logger.error("Validation failed! See errors above.")
    else:
        logger.info("Validation passed.")
        
    return report
