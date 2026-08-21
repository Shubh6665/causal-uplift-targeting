import pytest
import pandas as pd
import numpy as np
from src.data.validate import (
    validate_schema, validate_treatment, validate_outcome, 
    validate_no_leakage, run_all_validations, ValidationReport
)

@pytest.fixture
def mock_config():
    return {
        'dataset': {
            'criteo': {
                'feature_cols': [f'f{i}' for i in range(12)],
                'treatment_col': 'treatment',
                'outcome_col': 'conversion'
            }
        }
    }

@pytest.fixture
def valid_df():
    np.random.seed(42)
    data = {f'f{i}': np.random.rand(100) for i in range(12)}
    data['treatment'] = np.random.choice([0, 1], size=100)
    data['conversion'] = np.random.choice([0, 1], size=100)
    # Don't add exposure to keep it valid
    return pd.DataFrame(data)

def test_schema_valid(valid_df, mock_config):
    report = ValidationReport()
    validate_schema(valid_df, mock_config, report)
    assert report.is_valid
    assert len(report.errors) == 0

def test_schema_invalid_missing_col(valid_df, mock_config):
    report = ValidationReport()
    invalid_df = valid_df.drop(columns=['f0'])
    validate_schema(invalid_df, mock_config, report)
    assert not report.is_valid
    assert any('Missing features' in e for e in report.errors)

def test_treatment_invalid(valid_df):
    report = ValidationReport()
    invalid_df = valid_df.copy()
    invalid_df['treatment'] = np.random.choice([0, 1, 2], size=100) # Contains non-binary
    validate_treatment(invalid_df, 'treatment', report)
    assert not report.is_valid
    assert any('non-binary' in e for e in report.errors)

def test_no_leakage_invalid(valid_df, mock_config):
    report = ValidationReport()
    features = mock_config['dataset']['criteo']['feature_cols'] + ['exposure']
    validate_no_leakage(valid_df, features, report)
    assert not report.is_valid
    assert any('exposure' in e for e in report.errors)

def test_run_all_validations_passes(valid_df, mock_config):
    report = run_all_validations(valid_df, mock_config, dataset_type='criteo')
    # Because treatment ratio and conversion rate will be completely off for random data,
    # it might generate warnings, but the schema should be "valid" (no errors)
    assert report.is_valid
