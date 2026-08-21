import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def extract_X_T_Y(df: pd.DataFrame, config: dict, dataset_type: str = 'criteo') -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extract Features, Treatment, and Outcome columns."""
    if dataset_type == 'criteo':
        c_config = config['dataset']['criteo']
        X = df[c_config['feature_cols']]
        T = df[c_config['treatment_col']]
        Y = df[c_config['outcome_col']]
    elif dataset_type == 'ihdp':
        X = df[[f'x{i}' for i in range(1, 26)]]
        T = df['treatment']
        Y = df['y_factual']
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
    return X, T, Y

def create_splits(X: pd.DataFrame, T: pd.Series, Y: pd.Series, config: dict) -> dict:
    """
    Create train, validation, and frozen test splits.
    Returns: dict with keys 'train', 'val', 'test' containing (X, T, Y) tuples.
    """
    s_config = config['splitting']
    test_size = s_config['test_size']
    val_size = s_config['val_size']
    seed = s_config['seed']
    
    # Calculate relative validation size
    train_val_size = 1.0 - test_size
    rel_val_size = val_size / train_val_size
    
    # Try stratification levels: T+Y → T only → None
    # Small datasets (IHDP: 747 rows) can have singleton classes with T+Y stratification
    strat_array = None
    try:
        strat_df = pd.DataFrame({'T': T.values, 'Y': Y.values})
        strat_array = strat_df.astype(str).agg('-'.join, axis=1)
        # Check if any class has <2 members
        min_class_size = strat_array.value_counts().min()
        if min_class_size < 2:
            logger.info("T+Y stratification has singleton class, falling back to T-only.")
            strat_array = T.astype(str)
            min_class_size = strat_array.value_counts().min()
            if min_class_size < 2:
                logger.info("T-only stratification has singleton class, falling back to no stratification.")
                strat_array = None
    except Exception:
        strat_array = None

    logger.info("Splitting data into Train / Val / Test (Test is FROZEN).")
    
    # First split: (Train + Val) and Test
    if strat_array is not None:
        X_tv, X_test, T_tv, T_test, Y_tv, Y_test, strat_tv, _ = train_test_split(
            X, T, Y, strat_array, test_size=test_size, random_state=seed, stratify=strat_array
        )
    else:
        X_tv, X_test, T_tv, T_test, Y_tv, Y_test = train_test_split(
            X, T, Y, test_size=test_size, random_state=seed
        )
        strat_tv = None
    
    # Second split: Train and Val
    if strat_tv is not None:
        X_train, X_val, T_train, T_val, Y_train, Y_val = train_test_split(
            X_tv, T_tv, Y_tv, test_size=rel_val_size, random_state=seed, stratify=strat_tv
        )
    else:
        X_train, X_val, T_train, T_val, Y_train, Y_val = train_test_split(
            X_tv, T_tv, Y_tv, test_size=rel_val_size, random_state=seed
        )
    
    return {
        'train': (X_train, T_train, Y_train),
        'val': (X_val, T_val, Y_val),
        'test': (X_test, T_test, Y_test)
    }

def create_synthetic_dgp(
    n: int = 50000, 
    n_features: int = 20, 
    alpha: float = 1.0, 
    treatment_prob: float = 0.5, 
    seed: int = 42, 
    add_hidden_confounder: bool = False,
    confounder_strength: float = 1.0,
    X_fixed: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create Synthetic DGP for controlled stress tests.
    
    Args:
        alpha: Controls overlap. e(x) = sigmoid(alpha * X @ w). High alpha = weak overlap.
        treatment_prob: Base probability of treatment (for E2 Imbalance).
        add_hidden_confounder: If True, adds latent U influencing both T and Y (for E5).
        X_fixed: If provided, uses these features instead of generating new ones (useful for holding population constant in E2).
        
    Returns:
        X, T, Y, tau_true, prop_true
    """
    rng = np.random.default_rng(seed)
    
    # 1. Generate X or use fixed
    if X_fixed is not None:
        X = X_fixed
        n = len(X)
    else:
        X = rng.normal(0, 1, size=(n, n_features))
        
    # 2. Hidden Confounder U (if applicable)
    U = rng.normal(0, 1, size=n) if add_hidden_confounder else np.zeros(n)
        
    # 3. Propensity and Treatment Assignment
    # We want base probability to be `treatment_prob`.
    # e(x) = sigmoid(alpha * (X @ w) + bias + confounder_strength * U)
    w_e = rng.uniform(-1, 1, size=n_features)
    
    # Calculate raw logits without bias
    logits_raw = alpha * (X @ w_e) + (confounder_strength * U)
    
    # Find bias to achieve the desired average treatment probability approximately
    # Since we want E[T] = treatment_prob, we shift the logits.
    # Inverse sigmoid of treatment_prob as base bias
    base_bias = np.log(treatment_prob / (1.0 - treatment_prob)) if 0 < treatment_prob < 1 else 0.0
    
    logits = logits_raw - np.mean(logits_raw) + base_bias
    
    def sigmoid(z): return 1 / (1 + np.exp(-z))
    
    prop_true = sigmoid(logits)
    
    # Generate actual treatments
    T = rng.binomial(1, prop_true)
    
    # 4. Potential Outcomes
    # Y(0) = X @ w0 + U + noise
    # Y(1) = X @ w1 + U + noise
    # tau_true = Y(1) - Y(0) = X @ (w1 - w0)
    w0 = rng.uniform(-1, 1, size=n_features)
    w1 = w0 + rng.uniform(-0.5, 0.5, size=n_features)  # Heterogeneous treatment effect
    
    noise0 = rng.normal(0, 0.1, size=n)
    noise1 = rng.normal(0, 0.1, size=n)
    
    Y0 = X @ w0 + (confounder_strength * U) + noise0
    Y1 = X @ w1 + (confounder_strength * U) + noise1
    
    # Final observed outcome
    Y = T * Y1 + (1 - T) * Y0
    
    # True causal effect (ignoring noise)
    tau_true = X @ (w1 - w0)
    
    # Convert to pandas to match standard pipeline inputs
    X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(n_features)])
    T_series = pd.Series(T, name='treatment')
    Y_series = pd.Series(Y, name='outcome')
    
    return X_df, T_series, Y_series, tau_true, prop_true
