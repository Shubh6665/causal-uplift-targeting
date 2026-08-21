import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.causal.s_learner import SLearner
from src.causal.x_learner import XLearner
from src.evaluation.outcome_prediction import (
    compute_logloss, compute_pr_auc, compute_factual_metrics
)

def test_compute_logloss():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2])
    loss = compute_logloss(y_true, y_prob)
    assert loss > 0
    
    # Perfect predictions
    y_prob_perf = np.array([1.0, 0.0, 1.0, 0.0])
    # log_loss of perfect predictions should be very close to 0
    assert np.isclose(compute_logloss(y_true, y_prob_perf), 0, atol=1e-5)

def test_factual_metrics_with_s_learner():
    X = pd.DataFrame(np.random.rand(10, 2), columns=['A', 'B'])
    T = np.random.binomial(1, 0.5, size=10)
    Y = np.random.binomial(1, 0.5, size=10)
    
    model = SLearner(base_learner_class=LogisticRegression, base_learner_params={})
    model.fit(X, T, Y)
    
    metrics = compute_factual_metrics(X, T, Y, model)
    assert 'logloss' in metrics
    assert 'pr_auc' in metrics
    assert 'brier_score' in metrics

def test_factual_metrics_raises_for_x_learner():
    from sklearn.linear_model import LinearRegression
    X = pd.DataFrame(np.random.rand(10, 2), columns=['A', 'B'])
    T = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    Y = np.random.binomial(1, 0.5, size=10)
    
    prop_config = {'assignment_type': 'constant', 'known_probability': 0.5}
    # For XLearner on binary Y, base is classifier, effect is regressor
    model = XLearner(base_learner_class=LogisticRegression, base_learner_params={},
                     effect_learner_class=LinearRegression, effect_learner_params={})
    model.fit(X, T, Y, propensity_config=prop_config)
    
    with pytest.raises(ValueError, match="XLearner cannot directly predict"):
        compute_factual_metrics(X, T, Y, model)
