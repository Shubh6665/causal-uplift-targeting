import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from src.causal.s_learner import SLearner
from src.causal.t_learner import TLearner
from src.causal.x_learner import XLearner
from src.causal.propensity import get_propensity

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(np.random.rand(n, 3), columns=['f1', 'f2', 'f3'])
    T = np.random.binomial(1, 0.5, size=n)
    
    # Y = X*w + T*tau + noise
    # tau = 2.0 (constant effect for simplicity)
    Y = X['f1']*2 + T*2.0 + np.random.normal(0, 0.1, size=n)
    
    return X, T, Y

def test_s_learner(dummy_data):
    X, T, Y = dummy_data
    # Use simple LinearRegression as base learner
    model = SLearner(base_learner_class=LinearRegression, base_learner_params={})
    model.fit(X, T, Y)
    
    tau_hat = model.predict_tau(X)
    assert len(tau_hat) == len(X)
    # The effect should be close to 2.0
    assert np.isclose(np.mean(tau_hat), 2.0, atol=0.2)
    
    # Test predict_factual
    y_pred = model.predict_factual(X, T)
    assert len(y_pred) == len(Y)

def test_t_learner(dummy_data):
    X, T, Y = dummy_data
    model = TLearner(base_learner_class=LinearRegression, base_learner_params={})
    model.fit(X, T, Y)
    
    tau_hat = model.predict_tau(X)
    assert len(tau_hat) == len(X)
    assert np.isclose(np.mean(tau_hat), 2.0, atol=0.2)
    
    y_pred = model.predict_factual(X, T)
    assert len(y_pred) == len(Y)

def test_x_learner(dummy_data):
    X, T, Y = dummy_data
    
    # Need propensity for X-Learner prediction
    prop_config = {'assignment_type': 'constant', 'known_probability': 0.5}
    
    # LinearRegression for outcomes, and for effects
    model = XLearner(base_learner_class=LinearRegression, base_learner_params={})
    model.fit(X, T, Y, propensity_config=prop_config)
    
    tau_hat = model.predict_tau(X)
    assert len(tau_hat) == len(X)
    assert np.isclose(np.mean(tau_hat), 2.0, atol=0.3)
    
    D1, D0 = model.get_pseudo_effects()
    assert D1 is not None and len(D1) == sum(T == 1)
    assert D0 is not None and len(D0) == sum(T == 0)

def test_propensity_modes(dummy_data):
    X, T, Y = dummy_data
    
    # 1. Constant
    p_const = get_propensity(X, T, 'constant', known_probability=0.85)
    assert len(p_const) == len(X)
    assert np.all(p_const == 0.85)
    
    # 2. Estimated
    p_est = get_propensity(X, T, 'estimated', learner_class=LogisticRegression)
    assert len(p_est) == len(X)
    assert np.all((p_est >= 0) & (p_est <= 1))
    
    # 3. True known
    true_p = np.random.uniform(0.1, 0.9, size=len(X))
    p_true = get_propensity(X, T, 'true_known', true_propensities=true_p)
    assert np.array_equal(p_true, true_p)
