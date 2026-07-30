"""
Data Preparation Module

Handles dataset configuration, loading, and splitting for time series forecasting.
Designed to be reused across the thesis notebooks (classical, ML, and DL).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

from thesis.reporting import metadata_for

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================


def _calendar_year_splits() -> Dict:
    """Common frozen point-forecast chronology for the manual datasets."""
    return {
        'train': lambda df: df.index.year <= 2023,
        'val': lambda df: df.index.year == 2024,
        'test': lambda df: df.index.year == 2025,
    }


def _manual_dataset_config(
    *,
    label: str,
    file: str,
    target_raw: str,
    series_name: str,
    freq: str,
    seasonality: int,
    ml_lags,
    input_size: int,
    input_size_candidates,
    val_size: int,
) -> Dict:
    """Build one standardized univariate manual-dataset configuration."""
    return {
        'label': label,
        'file': file,
        'columns': {
            'id': 'unique_id',
            'time': 'ds',
            'target': 'y',
            'time_raw': 'indice_tiempo',
            'target_raw': target_raw,
        },
        'series_name': series_name,
        'freq': freq,
        'seasonality': int(seasonality),
        'selection_metric': 'rmse',
        'ml_lags': list(ml_lags),
        'dl': {
            'input_size': int(input_size),
            'input_size_candidates': list(input_size_candidates),
            'val_size': int(val_size),
            'auto_num_samples': 4,
            'auto_tpe_startup_trials': 2,
            # A common ceiling; validation-based early stopping determines the
            # effective duration for each fitted architecture.
            'auto_max_steps': 1000,
        },
        'splits': _calendar_year_splits(),
        'yearfirst': True,
        'is_synthetic': False,
    }


_MONTHLY_MANUAL_SETTINGS = {
    'freq': 'MS',
    'seasonality': 12,
    'ml_lags': list(range(1, 13)) + [24],
    'input_size': 24,
    'input_size_candidates': [12, 24, 36, 48],
    'val_size': 12,
}

_QUARTERLY_MANUAL_SETTINGS = {
    'freq': 'QS',
    'seasonality': 4,
    'ml_lags': [1, 2, 3, 4, 8],
    'input_size': 8,
    'input_size_candidates': [4, 8, 12, 16],
    'val_size': 4,
}


MANUAL_DATASETS = {
    'emae': _manual_dataset_config(
        label='EMAE (Argentina)',
        file='thesis/data/emae-valores-anuales-indice-base-2004-mensual.csv',
        target_raw='emae_original',
        series_name='EMAE',
        **_MONTHLY_MANUAL_SETTINGS,
    ),
    'ecai_us': _manual_dataset_config(
        label='Coincident Economic Activity Index (United States)',
        file='thesis/data/ecai_us_monthly.csv',
        target_raw='ecai',
        series_name='ECAI_US',
        **_MONTHLY_MANUAL_SETTINGS,
    ),
    'unemployment_us': _manual_dataset_config(
        label='Unemployment rate (United States)',
        file='thesis/data/unemployment_us_monthly.csv',
        target_raw='unemployment_rate',
        series_name='UNEMPLOYMENT_US',
        **_MONTHLY_MANUAL_SETTINGS,
    ),
    'real_gdp_argentina': _manual_dataset_config(
        label='Real GDP (Argentina)',
        file='thesis/data/real_gdp_argentina_quarterly.csv',
        target_raw='real_gdp',
        series_name='REAL_GDP_ARGENTINA',
        **_QUARTERLY_MANUAL_SETTINGS,
    ),
    'real_gdp_us': _manual_dataset_config(
        label='Real GDP (United States)',
        file='thesis/data/real_gdp_us_quarterly.csv',
        target_raw='real_gdp',
        series_name='REAL_GDP_US',
        **_QUARTERLY_MANUAL_SETTINGS,
    ),
}

# Manual datasets used by the three ordinary point-forecast notebooks.
# Unemployment is deliberately reserved for predictable_series_conformal.ipynb;
# the S&P 500, conformal AR(1), and conformal sinusoid use separate workflows.
POINT_FORECAST_DATASETS = (
    "emae",
    "ecai_us",
    "real_gdp_argentina",
    "real_gdp_us",
)

for _dataset_name, _dataset_config in MANUAL_DATASETS.items():
    _dataset_config["reporting"] = dict(metadata_for(_dataset_name))


# These files are prepared by data_wrangling.ipynb but intentionally excluded
# from MANUAL_DATASETS so batch point-forecast notebooks do not launch long
# daily DL experiments. Their conformal calibration/test chronology will be
# declared when the probabilistic experiment is finalized.
FINANCIAL_DATASETS = {
    'sp500_returns': {
        'label': 'S&P 500 daily log returns',
        'file': 'thesis/data/sp500_daily.csv',
        'time_raw': 'indice_tiempo',
        'target_raw': 'log_return',
        # Exchange holidays make pandas ``B`` irregular. The probabilistic
        # workflow will map observed trading dates to an integer ordinal.
        'freq': 1,
        'calendar': 'observed_trading_days',
        'probabilistic_only': True,
    },
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_manual_dataset(dataset_config: Dict) -> pd.DataFrame:
    """
    Load a manual dataset from CSV file.
    
    Parameters:
    -----------
    dataset_config : Dict
        Configuration dictionary with file path and column mappings
        
    Returns:
    --------
    pd.DataFrame
        Raw dataframe with datetime index
    """
    df = pd.read_csv(dataset_config['file'])
    
    # Parse datetime column
    time_raw = dataset_config['columns']['time_raw']
    df[time_raw] = pd.to_datetime(
        df[time_raw], 
        yearfirst=dataset_config.get('yearfirst', False)
    )
    df.set_index(time_raw, inplace=True)
    
    return df


def load_dataset(dataset_name: str, dataset_type: str = 'manual') -> Tuple[pd.DataFrame, Dict]:
    """
    Load dataset and return dataframe + config.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (key in MANUAL_DATASETS)
    dataset_type : str
        Type of dataset ('manual' or 'synthetic')
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict]
        Loaded dataframe and dataset configuration
    """
    if dataset_type != 'manual':
        raise ValueError(
            "Only manual splits are created here. Use split_global_panel "
            "from thesis.global_evaluation for synthetic data."
        )

    if dataset_type == 'manual':
        if dataset_name not in MANUAL_DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not found in MANUAL_DATASETS")
        config = MANUAL_DATASETS[dataset_name]
        df = load_manual_dataset(config)
    else:
        raise ValueError(
            "Only manual datasets are loaded here. Use "
            "thesis.global_evaluation.load_global_panel for synthetic data."
        )
    
    return df, config


# ============================================================================
# DATA SPLITTING FUNCTIONS
# ============================================================================

def create_train_val_test_splits(
    df: pd.DataFrame,
    dataset_config: Dict,
    dataset_type: str = 'manual'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str, str]:
    """
    Create train/val/test splits based on dataset configuration.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataframe
    dataset_config : Dict
        Configuration with split specifications
    dataset_type : str
        Type of dataset ('manual' or 'synthetic')
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str, str]
        (ts_train, ts_val, ts_test, id_col, time_col, target_col)
    """
    id_col = dataset_config['columns']['id']
    time_col = dataset_config['columns']['time']
    target_col = dataset_config['columns']['target']
    
    if dataset_type == 'manual':
        time_raw = dataset_config['columns']['time_raw']
        target_raw = dataset_config['columns']['target_raw']
        series_name = dataset_config.get('series_name', dataset_config['label'])
        
        # Apply split masks
        train_mask = dataset_config['splits']['train'](df)
        val_mask = dataset_config['splits']['val'](df)
        test_mask = dataset_config['splits']['test'](df)
        
        # Create split dataframes with standardized columns
        ts_train = df[train_mask][[target_raw]].reset_index().rename(
            columns={time_raw: time_col, target_raw: target_col}
        )
        ts_train[id_col] = series_name
        ts_train = ts_train[[id_col, time_col, target_col]]
        
        ts_val = df[val_mask][[target_raw]].reset_index().rename(
            columns={time_raw: time_col, target_raw: target_col}
        )
        ts_val[id_col] = series_name
        ts_val = ts_val[[id_col, time_col, target_col]]
        
        ts_test = df[test_mask][[target_raw]].reset_index().rename(
            columns={time_raw: time_col, target_raw: target_col}
        )
        ts_test[id_col] = series_name
        ts_test = ts_test[[id_col, time_col, target_col]]
        
    return ts_train, ts_val, ts_test, id_col, time_col, target_col


# ============================================================================
# RESULTS STORAGE FUNCTION
# ============================================================================

def initialize_results_dict(
    dataset_name: str,
    dataset_config: Dict,
    dataset_type: str,
    ts_train: pd.DataFrame,
    ts_val: pd.DataFrame,
    ts_test: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
    freq: str
) -> Dict:
    """
    Initialize the results dictionary for a dataset.
    
    Parameters:
    -----------
    Various dataset and split information
    
    Returns:
    --------
    Dict
        Results dictionary to be populated throughout the pipeline
    """
    return {
        'config': dataset_config,
        'type': dataset_type,
        'metrics': None,  # Will be filled by evaluate_performance
        'results': None,  # Will be filled by evaluate_performance
        'id_col': id_col,
        'time_col': time_col,
        'target_col': target_col,
        'freq': freq,
        'ts_train': ts_train,
        'ts_val': ts_val,
        'ts_test': ts_test,
        'pred_df': pred_df,
        'transformers': None,  # Will be filled by apply_stationarity_transform
        'transformed_data': None,  # Will be filled by apply_stationarity_transform
    }
