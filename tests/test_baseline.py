import pytest
import numpy as np
from src.causal.baseline import difference_in_means

def test_difference_in_means_toy():
    # Construct a toy dataset where ATE is known
    # T = [1, 1, 1, 0, 0, 0]
    # Y = [4, 5, 6, 1, 2, 3]
    # mean(Y|T=1) = 5
    # mean(Y|T=0) = 2
    # ATE_hat = 3
    
    T = np.array([1, 1, 1, 0, 0, 0])
    Y = np.array([4, 5, 6, 1, 2, 3])
    
    results = difference_in_means(Y, T)
    
    assert results['ate_hat'] == 3.0
    assert results['treated_mean'] == 5.0
    assert results['control_mean'] == 2.0
    assert results['treated_n'] == 3
    assert results['control_n'] == 3

def test_difference_in_means_empty():
    T = np.array([1, 1, 1])
    Y = np.array([4, 5, 6])
    
    # No control group
    results = difference_in_means(Y, T)
    assert results['control_mean'] == 0.0
    assert results['treated_mean'] == 5.0
    assert results['ate_hat'] == 5.0
