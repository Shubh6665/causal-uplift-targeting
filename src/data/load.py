import os
import pandas as pd
import numpy as np
import urllib.request
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_dev_subset(df: pd.DataFrame, frac: float, seed: int, stratify_cols: List[str]) -> pd.DataFrame:
    """Create a stratified subsample of the data."""
    if frac >= 1.0:
        return df
    
    # Fast stratified sampling using pandas groupby
    # For very large datasets, this might be memory intensive, but works well for Criteo
    logger.info(f"Creating {frac*100}% subset stratified by {stratify_cols}")
    subset = df.groupby(stratify_cols, group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=seed)
    )
    return subset

def load_criteo(config: dict, use_dev: bool = False) -> pd.DataFrame:
    """
    Load official Criteo Uplift v2.1 dataset.
    Reads the file using chunking to maintain memory safety.
    Extracts a reproducible subset to allow XGBoost to run efficiently.
    """
    c_config = config['dataset']['criteo']
    path = c_config['path']
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Official Criteo Uplift Modeling Dataset not found at {path}. Download the unbiased release before running benchmark experiments.")
        
    logger.info(f"Loading official Criteo dataset from {path}")
    
    # We use a 1% subset for dev, and a 5% subset for the final benchmark experiments
    # to avoid OOM / extreme training times locally. 5% is ~700,000 rows.
    frac = c_config['dev_sample_frac'] if use_dev else c_config['experiment_sample_frac']
    seed = c_config['dev_sample_seed']
    
    # Fast memory-safe subsetting using chunking
    chunks = []
    chunk_size = 500000
    
    logger.info(f"Subsampling {frac*100}% of Criteo data using chunked reading...")
    
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        # We perform the sampling at the chunk level
        sampled_chunk = chunk.sample(frac=frac, random_state=seed)
        chunks.append(sampled_chunk)
        
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Loaded a reproducible Criteo subset of {len(df)} rows.")
    
    return df

def load_ihdp(config: dict, simulation_id: int) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load a specific IHDP replication.
    Returns:
        X (DataFrame): Features
        tau_true (ndarray): True heterogeneous treatment effects (mu1 - mu0)
        propensity (ndarray): True propensity scores (not explicitly given in NPCI, we return None or estimate it later, but keeping signature for DGP parity)
    """
    ihdp_config = config['dataset']['ihdp']
    url = ihdp_config['source'].format(id=simulation_id)
    
    logger.info(f"Loading IHDP simulation {simulation_id} from {url}")
    
    # NPCI format has no header. 
    # Cols: treatment, y_factual, y_cfactual, mu0, mu1, x1..x25
    col_names = ['treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1'] + [f'x{i}' for i in range(1, 26)]
    
    try:
        df = pd.read_csv(url, header=None, names=col_names)
    except Exception as e:
        logger.error(f"Failed to load IHDP data from {url}: {e}")
        raise
        
    X = df[[f'x{i}' for i in range(1, 26)]]
    tau_true = (df['mu1'] - df['mu0']).values
    
    # NPCI dataset doesn't explicitly provide true propensities in this CSV format.
    # We return None, and it will be estimated in Phase 5.5 as per plan.
    propensity = None 
    
    # We also return the full df so it can be split into T, Y later if needed, 
    # or we just return the full DF and let preprocess extract X, T, Y.
    # To keep things clean for standard pipeline:
    return df, tau_true, propensity

def load_ihdp_multi(config: dict) -> List[Tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    """Load all configured IHDP replications."""
    sim_ids = config['dataset']['ihdp']['simulation_ids']
    results = []
    for sid in sim_ids:
        results.append(load_ihdp(config, sid))
    return results

def save_processed(df: pd.DataFrame, path: str):
    """Save processed dataframe to parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved processed data to {path}")

def load_processed(path: str) -> pd.DataFrame:
    """Load processed dataframe from parquet."""
    return pd.read_parquet(path)
