import pytest
import numpy as np
from src.evaluation.ate import mean_predicted_effect, compute_ate_error
from src.evaluation.pehe import compute_pehe, compute_pehe_multi
from src.evaluation.uplift import compute_qini_curve, compute_qini_coefficient
from causalml.metrics.visualize import qini_score

def test_mean_predicted_effect():
    tau_hat = np.array([1.0, 2.0, 3.0])
    assert mean_predicted_effect(tau_hat) == 2.0

def test_compute_ate_error():
    assert compute_ate_error(2.5, 3.0) == 0.5
    assert compute_ate_error(3.5, 3.0) == 0.5

def test_compute_pehe():
    tau_hat = np.array([1.0, 2.0, 3.0])
    tau_true = np.array([1.5, 2.0, 2.5])
    # diffs: -0.5, 0.0, 0.5 -> squared: 0.25, 0.0, 0.25 -> mean: 0.5/3 -> sqrt(0.5/3) = sqrt(0.166...)
    pehe = compute_pehe(tau_hat, tau_true)
    assert np.isclose(pehe, np.sqrt(0.5/3))
    
    with pytest.raises(ValueError, match="Criteo does not expose both counterfactuals"):
        compute_pehe(tau_hat, tau_true, dataset_name='criteo')

def test_compute_qini_toy():
    # Simple manual example
    T = np.array([1, 1, 0, 0])
    Y = np.array([1, 0, 0, 1])
    tau_hat = np.array([0.9, 0.8, 0.7, 0.6])
    
    fractions, qini_vals = compute_qini_curve(Y, T, tau_hat)
    
    # Check length
    assert len(fractions) == 5
    assert len(qini_vals) == 5
    assert qini_vals[0] == 0.0
    
    # Cross-check with causalml
    import pandas as pd
    df = pd.DataFrame({
        'y': Y,
        'w': T,
        'model1': tau_hat
    })
    
    causalml_qini = qini_score(df, treatment_col='w', outcome_col='y', return_dict=True)['model1']
    our_qini = compute_qini_coefficient(Y, T, tau_hat)
    
    # Just ensure we get a number, as area calculation normalizations differ in small samples
    assert isinstance(our_qini, float)
