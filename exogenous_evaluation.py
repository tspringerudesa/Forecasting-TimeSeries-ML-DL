"""Leakage-aware helpers for the Forte (2024) exogenous-variable benchmark."""

from __future__ import annotations

import contextlib
import copy
import gc
import itertools
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from thesis.runtime import CPU_THREADS, neuralforecast_compute_parameters


SEED = 67
DL_PROTOCOL = "rolling_h1_fixed_weights"
TREE_FEATURE_ELIGIBILITY = "origin_fit"
TARGET = "u01_arg_cpi_inflation_mom_pct"
_QUIET_STREAM = open(os.devnull, "w", encoding="utf-8")
SNAPSHOT_DIR = Path(
    "thesis/forte_2024_replication/data/processed/snapshots/20260725_final_v10"
)

VARIABLE_COLUMNS = {
    "U01": TARGET,
    "U02": "u02_registered_wage_growth_mom_pct",
    "U03": "u03_activity_growth_3mma_pct",
    "U04": "u04_nominal_interest_tem_pct",
    "U05": "u05_monetary_base_growth_mom_pct",
    "U06": "u06_m2_growth_mom_pct",
    "U07": "u07_wheat_price_growth_mom_pct",
    "U08": "u08_brent_price_growth_mom_pct",
    "U09": "u09_us_cpi_inflation_mom_pct",
    "U10": "u10_official_fx_growth_mom_pct",
    "U11": "u11_parallel_fx_growth_mom_pct",
    "U12": "u12_fx_gap_pct",
    "U13": "u13_real_net_reserves_mjul2024_usd",
}

FORTE_PROFILES = {
    "core8_long": {
        "file": "forte_core8_continuous_complete_monthly.csv",
        "label": "Core 8 long sample",
        "tuning_group": "core8_long",
        "native_missing_only": False,
    },
    "full13_complete_no_nir": {
        "file": "forte_full13_complete_monthly.csv",
        "label": "Full 13 complete sample without NIR",
        "drop_variables": ["U13"],
        "tuning_group": "full13_complete_no_nir",
        "native_missing_only": False,
    },
    "full_public_native": {
        "file": "forte_13_public_proxy_monthly_1965_2024.csv",
        "label": "Full public panel with native missing values",
        "tuning_group": "full_public_native",
        "native_missing_only": True,
    },
}

PRIMARY_EXOGENOUS_DL_MODELS = (
    "AutoRNN",
    "AutoTFT",
    "AutoNBEATSx",
    "AutoTimesNet",
    "AutoKAN",
    "AutoTiDE",
)


@contextlib.contextmanager
def quiet_neuralforecast_output():
    """Suppress library chatter while preserving notebook-level progress."""
    import optuna

    loggers = [
        logging.getLogger("lightning"),
        logging.getLogger("lightning.pytorch"),
        logging.getLogger("lightning_fabric"),
        logging.getLogger("pytorch_lightning"),
    ]
    previous_levels = [logger.level for logger in loggers]
    previous_optuna_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    try:
        with contextlib.redirect_stdout(
            _QUIET_STREAM
        ), contextlib.redirect_stderr(_QUIET_STREAM):
            yield
    finally:
        optuna.logging.set_verbosity(previous_optuna_verbosity)
        for logger, previous_level in zip(loggers, previous_levels):
            logger.setLevel(previous_level)

# These are models already used elsewhere in the thesis whose installed
# NeuralForecast classes expose future exogenous inputs. The primary subset
# has one representative from each recurrent, attention, basis-expansion,
# convolutional, KAN, and MLP family. The remaining candidates can be enabled
# from the notebook without changing this module.
DL_CONFIG_FILES = {
    "AutoTFT": "AutoTFT_config.json",
    "AutoNBEATSx": "AutoNBEATS_config.json",
    "AutoTimesNet": "AutoTimesNet_config.json",
    "AutoKAN": "AutoKAN_config.json",
    "AutoTiDE": "AutoTiDE_config.json",
    "AutoTSMixerx": "AutoTSMixer_config.json",
    "AutoRNN": "AutoRNN_config.json",
    "AutoLSTM": "AutoLSTM_config.json",
    "AutoGRU": "AutoGRU_config.json",
    "AutoVanillaTransformer": "AutoVanillaTransformer_config.json",
    "AutoAutoformer": "AutoAutoformer_config.json",
    "AutoNHITS": "AutoNHITS_config.json",
}


def economic_feature_group(feature: str) -> str:
    """Map lag, missingness, and quality fields to an economic driver."""
    normalized = feature.removeprefix("missing__").lower()
    quality_groups = {
        "q_cpi": "Argentine inflation",
        "q_wage": "Registered wages",
        "q_pre_comparable_brent": "Brent oil",
        "q_unified_fx": "Exchange-rate regime",
        "q_parallel_source": "Parallel exchange rate",
        "q_nir": "Net international reserves",
        "q_u04": "Nominal interest rate",
    }
    for prefix, group in quality_groups.items():
        if normalized.startswith(prefix):
            return group
    variable_groups = (
        (("us_cpi", "u09_", "f13_"), "US inflation"),
        (("inflation_lag", "arg_cpi", "u01_", "f01_", "f02_"), "Argentine inflation"),
        (("registered_wage", "wage", "u02_", "f05_"), "Registered wages"),
        (("activity", "emae", "u03_", "f10_"), "Economic activity"),
        (
            ("nominal_interest", "interest", "u04_", "f03_"),
            "Nominal interest rate",
        ),
        (("monetary_base", "u05_"), "Monetary base"),
        (("m2_", "u06_", "f04_"), "M2"),
        (("wheat", "u07_", "f15_"), "Wheat price"),
        (("brent", "u08_", "f14_"), "Brent oil"),
        (("official_fx", "u10_", "f08_", "f09_"), "Official exchange rate"),
        (("parallel_fx", "u11_", "f06_", "f07_"), "Parallel exchange rate"),
        (("fx_gap", "u12_", "f12_"), "Exchange-rate gap"),
        (
            ("real_net_reserves", "nir_", "u13_", "f11_"),
            "Net international reserves",
        ),
    )
    for tokens, group in variable_groups:
        if any(token in normalized for token in tokens):
            return group
    return "Other construction indicator"


def economic_feature_label(feature: str) -> str:
    """Return a publication-friendly label for one engineered predictor."""
    missing_indicator = feature.startswith("missing__")
    normalized = feature.removeprefix("missing__").lower()
    quality_labels = {
        "q_cpi_reconstruction_period": "Argentine CPI reconstruction period",
        "q_wage_isbic_regime_lag1": "ISBIC wage-source regime (t−1)",
        "q_wage_ripte_regime_lag1": "RIPTE wage-source regime (t−1)",
        "q_wage_source_transition_lag1": "Wage-source transition (t−1)",
        "q_pre_comparable_brent_lag1": "Pre-comparable Brent period (t−1)",
        "q_u04_suspect_zero_lag1": "Suspect interest-rate zero (t−1)",
        "q_unified_fx_market_lag1": "Unified FX-market regime (t−1)",
        "q_parallel_source_transition_lag1": (
            "Parallel-FX source transition (t−1)"
        ),
        "q_nir_blank_assumption_lag1": "NIR blank-liability assumption (t−1)",
    }
    if normalized in quality_labels:
        return quality_labels[normalized]

    variable_labels = (
        (("inflation_lag", "u01_", "f01_", "f02_"), "Argentine CPI inflation"),
        (("u02_", "f05_"), "Registered wage growth"),
        (("u03_", "f10_"), "Economic activity growth"),
        (("u04_", "f03_"), "Nominal interest rate"),
        (("u05_",), "Monetary base growth"),
        (("u06_", "f04_"), "M2 growth"),
        (("u07_", "f15_"), "Wheat-price growth"),
        (("u08_", "f14_"), "Brent-price growth"),
        (("u09_", "f13_"), "US CPI inflation"),
        (("u10_", "f08_", "f09_"), "Official exchange-rate growth"),
        (("u11_", "f06_", "f07_"), "Parallel exchange-rate growth"),
        (("u12_", "f12_"), "Exchange-rate gap"),
        (("u13_", "f11_"), "Real net international reserves"),
    )
    label = economic_feature_group(normalized)
    for tokens, candidate in variable_labels:
        if any(token in normalized for token in tokens):
            label = candidate
            break

    lag_match = re.search(r"(?:lag_|lag)(\d+)$", normalized)
    if lag_match:
        label = f"{label} (t−{int(lag_match.group(1))})"
    elif normalized.endswith("_current"):
        label = f"{label} (t)"
    if missing_indicator:
        label = f"Missing indicator: {label}"
    return label


def _read_monthly(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.sort_values("date").reset_index(drop=True)


def audit_missing_and_zeros(
    snapshot_dir: Path = SNAPSHOT_DIR,
) -> pd.DataFrame:
    """Report missing and exact-zero counts without declaring every zero invalid."""
    frame = _read_monthly(
        Path(snapshot_dir) / "forte_13_public_proxy_monthly_1965_2024.csv"
    )
    rows = []
    for column in frame.columns:
        if column == "date":
            continue
        values = frame[column]
        zero_dates = frame.loc[values.eq(0), "date"]
        rows.append(
            {
                "Variable": column,
                "Missing": int(values.isna().sum()),
                "Exact Zeros": int(values.eq(0).sum()),
                "First Zero": (
                    zero_dates.min().date().isoformat()
                    if not zero_dates.empty
                    else None
                ),
                "Last Zero": (
                    zero_dates.max().date().isoformat()
                    if not zero_dates.empty
                    else None
                ),
                "Minimum": float(values.min(skipna=True)),
                "Maximum": float(values.max(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def _construction_quality_flags(
    details: pd.DataFrame,
) -> pd.DataFrame:
    """Build fixed semantic flags; no data-driven encoding is learned globally."""
    indexed = details.set_index("date").sort_index()

    def bool_flag(values: pd.Series) -> pd.Series:
        return pd.Series(
            values.to_numpy(dtype=bool, na_value=False),
            index=values.index,
        )

    source = indexed["wages__registered_wage_source"].astype("string")
    wage_transition = bool_flag(source.ne(source.shift()) & source.notna())
    source_at_lag1 = source.shift(1)

    flags = pd.DataFrame(index=indexed.index)
    flags["q_cpi_reconstruction_period"] = indexed.index.to_series().between(
        pd.Timestamp("2007-01-01"), pd.Timestamp("2016-12-01")
    ).astype("int8")
    flags["q_wage_isbic_regime_lag1"] = (
        source_at_lag1.str.contains("ISBIC", case=False, na=False).astype("int8")
    )
    flags["q_wage_ripte_regime_lag1"] = (
        source_at_lag1.str.contains("RIPTE", case=False, na=False).astype("int8")
    )
    flags["q_wage_source_transition_lag1"] = (
        wage_transition.shift(1, fill_value=False).astype("int8")
    )
    source_month_lag1 = indexed.index.to_series().shift(1)
    flags["q_pre_comparable_brent_lag1"] = (
        source_month_lag1.lt(pd.Timestamp("1987-01-01")).fillna(False).astype("int8")
    )
    flags["q_u04_suspect_zero_lag1"] = (
        indexed["standard__nominal_interest_tem_pct"]
        .eq(0)
        .shift(1, fill_value=False)
        .astype("int8")
    )
    flags["q_unified_fx_market_lag1"] = (
        bool_flag(indexed["parallel_fx__fx_gap_imposed_zero_unified_flag"])
        .shift(1, fill_value=False)
        .astype("int8")
    )
    flags["q_parallel_source_transition_lag1"] = (
        bool_flag(indexed["parallel_fx__parallel_fx_source_transition_flag"])
        .shift(1, fill_value=False)
        .astype("int8")
    )
    flags["q_nir_blank_assumption_lag1"] = (
        bool_flag(indexed["nir__nir_blank_or_absent_assumption_flag"])
        .shift(1, fill_value=False)
        .astype("int8")
    )
    flags["q_unified_fx_market_current"] = (
        bool_flag(indexed["parallel_fx__fx_gap_imposed_zero_unified_flag"]).astype(
            "int8"
        )
    )
    flags["q_parallel_source_transition_current"] = (
        bool_flag(indexed["parallel_fx__parallel_fx_source_transition_flag"]).astype(
            "int8"
        )
    )
    return flags.reset_index()


def _strict_lagged_features(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    features = {
        "inflation_lag_1": frame[TARGET].shift(1),
        "inflation_lag_2": frame[TARGET].shift(2),
    }
    for variable_id, column in VARIABLE_COLUMNS.items():
        if variable_id == "U01" or column not in frame:
            continue
        lag = 2 if variable_id == "U03" else 1
        features[f"{variable_id.lower()}_lag_{lag}"] = frame[column].shift(lag)
    return features


def _forte_figure4_features(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    definitions = [
        ("f01_inflation_lag_1", "U01", 1),
        ("f02_inflation_lag_2", "U01", 2),
        ("f03_interest_lag_1", "U04", 1),
        ("f04_m2_lag_1", "U06", 1),
        ("f05_wages_lag_1", "U02", 1),
        ("f06_parallel_fx_lag_1", "U11", 1),
        ("f07_parallel_fx_current", "U11", 0),
        ("f08_official_fx_lag_1", "U10", 1),
        ("f09_official_fx_current", "U10", 0),
        ("f10_activity_lag_2", "U03", 2),
        ("f11_nir_lag_1", "U13", 1),
        ("f12_fx_gap_lag_1", "U12", 1),
        ("f13_us_cpi_lag_1", "U09", 1),
        ("f14_brent_lag_1", "U08", 1),
        ("f15_wheat_lag_1", "U07", 1),
    ]
    features = {}
    for feature_name, variable_id, lag in definitions:
        column = VARIABLE_COLUMNS[variable_id]
        if column in frame:
            features[feature_name] = frame[column].shift(lag)
    return features


def build_forte_profile(
    profile_name: str,
    snapshot_dir: Path = SNAPSHOT_DIR,
    feature_mode: str = "strict_lagged",
) -> Tuple[pd.DataFrame, Dict]:
    """Create one supervised one-month-ahead modelling frame."""
    if profile_name not in FORTE_PROFILES:
        raise ValueError(
            f"Unknown Forte profile {profile_name!r}; choose from "
            f"{sorted(FORTE_PROFILES)}"
        )
    config = dict(FORTE_PROFILES[profile_name])
    snapshot_dir = Path(snapshot_dir)
    frame = _read_monthly(snapshot_dir / config["file"])
    details = _read_monthly(
        snapshot_dir / "forte_13_construction_details_monthly.csv"
    )

    if config.get("start"):
        frame = frame.loc[frame["date"].ge(config["start"])].copy()
    for variable_id in config.get("drop_variables", []):
        frame = frame.drop(columns=[VARIABLE_COLUMNS[variable_id]])

    # U04 is an effective nominal rate. The 1982-07--1984-05 block of exact
    # zeros is economically implausible and is treated as missing. Genuine
    # zero growth rates and documented zero FX gaps are retained.
    if VARIABLE_COLUMNS["U04"] in frame:
        frame.loc[
            frame[VARIABLE_COLUMNS["U04"]].eq(0), VARIABLE_COLUMNS["U04"]
        ] = np.nan

    if feature_mode == "strict_lagged":
        features = _strict_lagged_features(frame)
    elif feature_mode == "forte_figure4_retrospective":
        features = _forte_figure4_features(frame)
    else:
        raise ValueError(
            "feature_mode must be 'strict_lagged' or "
            "'forte_figure4_retrospective'"
        )

    modelling = pd.DataFrame(
        {
            "date": frame["date"],
            "y": frame[TARGET].astype(float),
            **features,
        }
    )
    predictor_columns = list(features)
    for column in predictor_columns:
        modelling[f"missing__{column}"] = modelling[column].isna().astype("int8")

    flags = _construction_quality_flags(details)
    modelling = modelling.merge(flags, on="date", how="left", validate="one_to_one")
    if feature_mode == "strict_lagged":
        modelling = modelling.drop(
            columns=[
                "q_unified_fx_market_current",
                "q_parallel_source_transition_current",
            ]
        )
    if VARIABLE_COLUMNS["U13"] not in frame:
        modelling = modelling.drop(columns=["q_nir_blank_assumption_lag1"])
    if VARIABLE_COLUMNS["U04"] not in frame:
        modelling = modelling.drop(columns=["q_u04_suspect_zero_lag1"])
    if VARIABLE_COLUMNS["U11"] not in frame:
        modelling = modelling.drop(
            columns=[
                column
                for column in modelling
                if column.startswith("q_unified_fx")
                or column.startswith("q_parallel_source")
            ]
        )

    # The first two rows do not have the target history required by either
    # feature mode. Other NaNs are retained only for native-missing tree models.
    modelling = modelling.iloc[2:].reset_index(drop=True)
    feature_columns = [
        column for column in modelling.columns if column not in {"date", "y"}
    ]
    if not config["native_missing_only"]:
        missing = modelling[feature_columns].isna().sum()
        if int(missing.sum()) != 0:
            offenders = missing[missing.gt(0)].to_dict()
            raise ValueError(
                f"{profile_name} is intended to be complete but contains "
                f"feature NaNs: {offenders}"
            )

    config.update(
        {
            "profile": profile_name,
            "feature_mode": feature_mode,
            "feature_columns": feature_columns,
            "rows": len(modelling),
            "first_month": modelling["date"].min().date().isoformat(),
            "last_month": modelling["date"].max().date().isoformat(),
        }
    )
    return modelling, config


def split_forte_profile(
    frame: pd.DataFrame,
    validation_size: int = 12,
    test_size: int = 12,
) -> Dict[str, pd.DataFrame]:
    """Create aligned final validation and test blocks."""
    if min(validation_size, test_size) < 1:
        raise ValueError("validation_size and test_size must be positive")
    if len(frame) <= validation_size + test_size:
        raise ValueError("Forte profile is too short for the requested splits")
    train_end = len(frame) - validation_size - test_size
    validation_end = len(frame) - test_size
    return {
        "train": frame.iloc[:train_end].reset_index(drop=True),
        "validation": frame.iloc[train_end:validation_end].reset_index(drop=True),
        "test": frame.iloc[validation_end:].reset_index(drop=True),
    }


def fit_and_eval_frames(
    splits: Dict[str, pd.DataFrame],
    evaluation_split: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if evaluation_split == "validation":
        return splits["train"].copy(), splits["validation"].copy()
    if evaluation_split == "test":
        return (
            pd.concat([splits["train"], splits["validation"]], ignore_index=True),
            splits["test"].copy(),
        )
    raise ValueError("evaluation_split must be 'validation' or 'test'")


def active_features(
    fit_df: pd.DataFrame,
    candidate_features: Iterable[str],
) -> List[str]:
    """Fit-sample-only removal of constant or wholly missing regressors."""
    active = []
    for column in candidate_features:
        values = fit_df[column]
        if values.notna().sum() and values.nunique(dropna=True) > 1:
            active.append(column)
    if not active:
        raise ValueError("No varying exogenous features remain in the fitting sample")
    return active


def _target_lag_columns(feature_mode: str) -> Dict[str, int]:
    if feature_mode == "strict_lagged":
        return {"inflation_lag_1": 1, "inflation_lag_2": 2}
    if feature_mode == "forte_figure4_retrospective":
        return {"f01_inflation_lag_1": 1, "f02_inflation_lag_2": 2}
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def future_exogenous_features(
    feature_columns: Iterable[str],
    feature_mode: str = "strict_lagged",
) -> List[str]:
    """Remove target-derived columns that are unknown beyond forecast origin."""
    target_lags = set(_target_lag_columns(feature_mode))
    target_missing = {f"missing__{column}" for column in target_lags}
    return [
        column
        for column in feature_columns
        if column not in target_lags and column not in target_missing
    ]


def _auto_stationary_transformer():
    from src.transforms.target_transformations import AutoStationaryTransformer

    # Pass new dictionaries explicitly because the source class mutates some
    # parameter dictionaries while building its fitted pipeline.
    return AutoStationaryTransformer(
        seasonal_period=12,
        trend_check_params={"mann_kendall": False},
        detrender_params={"degree": 1},
        deseasonalizer_params={},
        box_cox_params={"optimization": "guerrero"},
    )


def prepare_tree_origin(
    origin_fit: pd.DataFrame,
    origin_eval: pd.DataFrame,
    feature_columns: List[str],
    target_transformation: str = "Original",
    feature_mode: str = "strict_lagged",
    freq: str = "MS",
):
    """Prepare one origin without allowing transformation leakage."""
    if target_transformation == "Original":
        return origin_fit.copy(), origin_eval.copy(), None
    if target_transformation != "AutoStationary":
        raise ValueError(
            "target_transformation must be 'Original' or 'AutoStationary'"
        )
    if len(origin_eval) != 1:
        raise ValueError(
            "AutoStationary tree evaluation requires one-step origins"
        )

    lag_columns = _target_lag_columns(feature_mode)
    missing_lags = set(lag_columns).difference(feature_columns)
    if missing_lags:
        raise ValueError(
            f"Transformed target lag columns are inactive: {sorted(missing_lags)}"
        )

    ordered_fit = origin_fit.sort_values("date").reset_index(drop=True)
    index = pd.DatetimeIndex(pd.to_datetime(ordered_fit["date"]))
    original = pd.Series(
        ordered_fit["y"].astype(float).to_numpy(),
        index=index,
        name="y",
    )
    transformer = _auto_stationary_transformer()
    transformed = transformer.fit_transform(original, freq=freq)
    transformed = pd.Series(
        np.asarray(transformed, dtype=float),
        index=index,
        name="y",
    )

    transformed_fit = ordered_fit.copy()
    transformed_fit["y"] = transformed.to_numpy()
    transformed_eval = origin_eval.copy()
    for column, lag in lag_columns.items():
        transformed_fit[column] = transformed.shift(lag).to_numpy()
        transformed_eval.loc[:, column] = float(transformed.iloc[-lag])
        missing_column = f"missing__{column}"
        if missing_column in transformed_fit:
            transformed_fit[missing_column] = (
                transformed_fit[column].isna().astype("int8")
            )
            transformed_eval.loc[:, missing_column] = 0

    transformed_fit = transformed_fit.dropna(
        subset=["y"] + list(lag_columns)
    ).reset_index(drop=True)
    if transformed_fit.empty:
        raise ValueError("No transformed fitting rows remain at this origin")
    return transformed_fit, transformed_eval, transformer


def prepare_tree_horizon(
    origin_fit: pd.DataFrame,
    origin_eval: pd.DataFrame,
    feature_columns: List[str],
    target_transformation: str = "Original",
    feature_mode: str = "strict_lagged",
    freq: str = "MS",
):
    """Prepare one multi-step tree origin without target or transform leakage."""
    ordered_fit = origin_fit.sort_values("date").reset_index(drop=True)
    ordered_eval = origin_eval.sort_values("date").reset_index(drop=True)
    if target_transformation == "Original":
        return ordered_fit, ordered_eval, None
    if target_transformation != "AutoStationary":
        raise ValueError(
            "target_transformation must be 'Original' or 'AutoStationary'"
        )

    lag_columns = _target_lag_columns(feature_mode)
    missing_lags = set(lag_columns).difference(feature_columns)
    if missing_lags:
        raise ValueError(
            f"Transformed target lag columns are inactive: {sorted(missing_lags)}"
        )
    index = pd.DatetimeIndex(pd.to_datetime(ordered_fit["date"]))
    original = pd.Series(
        ordered_fit["y"].astype(float).to_numpy(),
        index=index,
        name="y",
    )
    transformer = _auto_stationary_transformer()
    transformed = pd.Series(
        np.asarray(transformer.fit_transform(original, freq=freq), dtype=float),
        index=index,
        name="y",
    )
    transformed_fit = ordered_fit.copy()
    transformed_fit["y"] = transformed.to_numpy()
    for column, lag in lag_columns.items():
        transformed_fit[column] = transformed.shift(lag).to_numpy()
        missing_column = f"missing__{column}"
        if missing_column in transformed_fit:
            transformed_fit[missing_column] = (
                transformed_fit[column].isna().astype("int8")
            )
    transformed_fit = transformed_fit.dropna(
        subset=["y"] + list(lag_columns)
    ).reset_index(drop=True)
    if transformed_fit.empty:
        raise ValueError("No transformed fitting rows remain at this origin")
    return transformed_fit, ordered_eval, transformer


def recursive_tree_prediction(
    fitted,
    prepared_fit: pd.DataFrame,
    prepared_eval: pd.DataFrame,
    feature_columns: List[str],
    feature_mode: str = "strict_lagged",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Forecast a complete horizon while recursively rebuilding target lags."""
    lag_columns = _target_lag_columns(feature_mode)
    history = prepared_fit["y"].astype(float).tolist()
    predictions = []
    design_rows = []
    for _, evaluation_row in prepared_eval.iterrows():
        row = evaluation_row.copy()
        for column, lag in lag_columns.items():
            if column not in feature_columns:
                continue
            row[column] = float(history[-lag])
            missing_column = f"missing__{column}"
            if missing_column in feature_columns:
                row[missing_column] = 0
        design = pd.DataFrame([row], columns=prepared_eval.columns)
        prediction = float(fitted.predict(design[feature_columns])[0])
        predictions.append(prediction)
        history.append(prediction)
        design_rows.append(design.iloc[0])
    return (
        np.asarray(predictions, dtype=float),
        pd.DataFrame(design_rows).reset_index(drop=True),
    )


def inverse_tree_prediction(
    prediction: np.ndarray,
    eval_df: pd.DataFrame,
    transformer,
) -> np.ndarray:
    """Return transformed forecasts to the original inflation scale."""
    values = np.asarray(prediction, dtype=float).reshape(-1)
    if transformer is None:
        return values
    if len(values) != len(eval_df):
        raise ValueError("Forecast and evaluation horizons must have equal length")
    transformed = pd.Series(
        values,
        index=pd.DatetimeIndex(pd.to_datetime(eval_df["date"])),
        name="prediction",
    )
    return np.asarray(
        transformer.inverse_transform(transformed),
        dtype=float,
    ).reshape(-1)


def quality_weights(
    frame: pd.DataFrame,
    multipliers: Optional[Dict[str, float]] = None,
) -> Optional[np.ndarray]:
    """Optional tree-only quality weighting; disabled when no map is supplied."""
    if not multipliers:
        return None
    weights = np.ones(len(frame), dtype=float)
    for column, multiplier in multipliers.items():
        if column not in frame:
            raise ValueError(f"Unknown quality-weight column: {column}")
        if not 0 < float(multiplier) <= 1:
            raise ValueError("Quality multipliers must be in (0, 1]")
        weights *= np.where(frame[column].astype(bool), float(multiplier), 1.0)
    return weights


def _tree_estimator(
    model_name: str,
    params: Dict,
    seed: int = SEED,
    n_jobs: int = CPU_THREADS,
):
    if model_name == "XGBRandomForest":
        from xgboost import XGBRFRegressor

        fixed = {
            # XGBRF does not support reg:absoluteerror when num_parallel_tree
            # exceeds one. Its trees therefore retain squared-error fitting,
            # while Optuna, model selection, and reporting use MAE.
            "objective": "reg:squarederror",
            "learning_rate": 1.0,
            "tree_method": "hist",
            "missing": np.nan,
            "random_state": seed,
            "n_jobs": n_jobs,
            "verbosity": 0,
        }
        fixed.update(params)
        return XGBRFRegressor(**fixed)
    if model_name == "LightGBM":
        from lightgbm import LGBMRegressor

        fixed = {
            "objective": "regression_l1",
            "n_estimators": 6000,
            "random_state": seed,
            "n_jobs": n_jobs,
            "verbosity": -1,
        }
        fixed.update(params)
        return LGBMRegressor(**fixed)
    raise ValueError(f"Unsupported tree model: {model_name}")


def _fit_tree(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray],
    early_stopping_fraction: float = 0.2,
    early_stopping_rounds: int = 100,
):
    """Select boosting iterations chronologically, then refit all fitting rows."""
    from sklearn.base import clone

    fitted = clone(estimator)
    best_iteration = None
    if type(estimator).__name__ == "LGBMRegressor":
        from lightgbm import early_stopping

        validation_size = max(1, int(np.ceil(len(X) * early_stopping_fraction)))
        split_at = len(X) - validation_size
        if split_at < 2:
            raise ValueError("Too few rows for LightGBM temporal early stopping")
        stopping_model = clone(estimator)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[:split_at]
            fit_kwargs["eval_sample_weight"] = [sample_weight[split_at:]]
        stopping_model.fit(
            X.iloc[:split_at],
            y.iloc[:split_at],
            eval_set=[(X.iloc[split_at:], y.iloc[split_at:])],
            eval_metric="mae",
            callbacks=[
                early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
            ],
            **fit_kwargs,
        )
        best_iteration = max(1, int(stopping_model.best_iteration_))
        fitted.set_params(n_estimators=best_iteration)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    fitted.fit(X, y, **fit_kwargs)
    return fitted, best_iteration


def tune_tree_model(
    train_df: pd.DataFrame,
    feature_columns: List[str],
    model_name: str,
    n_trials: int = 25,
    n_splits: int = 3,
    fold_size: int = 12,
    seed: int = SEED,
    n_jobs: int = CPU_THREADS,
    weight_multipliers: Optional[Dict[str, float]] = None,
    recalibrate_each_origin: bool = True,
    target_transformation: str = "Original",
    feature_mode: str = "strict_lagged",
    freq: str = "MS",
) -> Dict:
    """Tune trees with expanding chronological holdouts."""
    import optuna
    from sklearn.model_selection import TimeSeriesSplit

    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=fold_size)
    def objective(trial):
        if model_name == "XGBRandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 0.5, 20, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bynode": trial.suggest_float(
                    "colsample_bynode", 0.5, 1.0
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-3, 20, log=True
                ),
                "gamma": trial.suggest_float("gamma", 0, 5),
            }
        elif model_name == "LightGBM":
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.005, 0.08, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 8, 96),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 5, 60
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-8, 10, log=True
                ),
            }
        else:
            raise ValueError(f"Unsupported tree model: {model_name}")
        fold_scores = []
        for train_index, validation_index in splitter.split(train_df):
            fold_predictions = []
            origin_steps = (
                range(len(validation_index))
                if recalibrate_each_origin
                else range(1)
            )
            for step in origin_steps:
                if recalibrate_each_origin:
                    origin_train_index = np.concatenate(
                        [train_index, validation_index[:step]]
                    )
                    origin_validation_index = validation_index[step : step + 1]
                else:
                    origin_train_index = train_index
                    origin_validation_index = validation_index
                origin_fit = train_df.iloc[origin_train_index].copy()
                origin_eval = train_df.iloc[origin_validation_index].copy()
                origin_features = active_features(
                    origin_fit, feature_columns
                )
                preparation = (
                    prepare_tree_origin
                    if recalibrate_each_origin
                    else prepare_tree_horizon
                )
                prepared_fit, prepared_eval, transformer = preparation(
                    origin_fit,
                    origin_eval,
                    origin_features,
                    target_transformation=target_transformation,
                    feature_mode=feature_mode,
                    freq=freq,
                )
                estimator = _tree_estimator(model_name, params, seed, n_jobs)
                fold_weights = quality_weights(
                    prepared_fit,
                    weight_multipliers,
                )
                fitted, _ = _fit_tree(
                    estimator,
                    prepared_fit[origin_features],
                    prepared_fit["y"],
                    fold_weights,
                )
                if recalibrate_each_origin:
                    transformed_prediction = fitted.predict(
                        prepared_eval[origin_features]
                    )
                else:
                    transformed_prediction, _ = recursive_tree_prediction(
                        fitted,
                        prepared_fit,
                        prepared_eval,
                        origin_features,
                        feature_mode=feature_mode,
                    )
                fold_predictions.extend(
                    inverse_tree_prediction(
                        transformed_prediction,
                        origin_eval,
                        transformer,
                    ).tolist()
                )
            actual = train_df.iloc[validation_index]["y"].to_numpy()
            fold_scores.append(
                float(
                    np.mean(
                        np.abs(actual - np.asarray(fold_predictions, dtype=float))
                    )
                )
            )
        return float(np.mean(fold_scores))

    start = time.perf_counter()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.perf_counter() - start
    return {
        "model": model_name,
        "parameters": study.best_params,
        "best_cv_mae": float(study.best_value),
        "tuning_seconds": elapsed,
        "n_trials": n_trials,
        "n_splits": n_splits,
        "fold_size": fold_size,
        "seed": seed,
        "weight_multipliers": weight_multipliers or {},
        "recalibrate_each_origin": recalibrate_each_origin,
        "target_transformation": target_transformation,
        "feature_mode": feature_mode,
        "feature_eligibility": TREE_FEATURE_ELIGIBILITY,
        "freq": freq,
    }


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    error = predicted - actual
    absolute_error = np.abs(error)
    denominator = abs(float(np.sum(actual)))
    bias = 0.0 if denominator == 0 else 100.0 * float(np.sum(error)) / denominator
    return {
        # Inflation is stored as a monthly percentage rate, so MAE, RMSE, and
        # this dispersion are measured in percentage points.
        "mae": float(np.mean(absolute_error)),
        "absolute_error_sd": (
            float(np.std(absolute_error, ddof=1))
            if len(absolute_error) > 1
            else 0.0
        ),
        "mse": float(np.mean(np.square(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "forecast_bias": bias,
    }


def evaluate_tree_model(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: List[str],
    config: Dict,
    n_jobs: int = CPU_THREADS,
    weight_multipliers: Optional[Dict[str, float]] = None,
    recalibrate_each_origin: bool = True,
    feature_mode: str = "strict_lagged",
    freq: str = "MS",
):
    model_name = config["model"]
    target_transformation = config.get(
        "target_transformation",
        "Original",
    )
    forecast_name = (
        model_name
        if target_transformation == "Original"
        else f"{model_name}_AutoStationary"
    )
    predictions = []
    training_time = 0.0
    forecast_time = 0.0
    fitted = None
    best_iteration = None
    origins = range(len(eval_df)) if recalibrate_each_origin else range(1)
    for origin in origins:
        origin_fit = (
            pd.concat([fit_df, eval_df.iloc[:origin]], ignore_index=True)
            if recalibrate_each_origin
            else fit_df
        )
        origin_eval = (
            eval_df.iloc[[origin]]
            if recalibrate_each_origin
            else eval_df
        )
        origin_features = active_features(origin_fit, feature_columns)
        preparation = (
            prepare_tree_origin
            if recalibrate_each_origin
            else prepare_tree_horizon
        )
        prepared_fit, prepared_eval, transformer = preparation(
            origin_fit,
            origin_eval,
            origin_features,
            target_transformation=target_transformation,
            feature_mode=feature_mode,
            freq=freq,
        )
        estimator = _tree_estimator(
            model_name,
            config["parameters"],
            seed=int(config.get("seed", SEED)),
            n_jobs=n_jobs,
        )
        weights = quality_weights(prepared_fit, weight_multipliers)
        start = time.perf_counter()
        fitted, best_iteration = _fit_tree(
            estimator,
            prepared_fit[origin_features],
            prepared_fit["y"],
            weights,
        )
        training_time += time.perf_counter() - start
        forecast_start = time.perf_counter()
        if recalibrate_each_origin:
            transformed_prediction = fitted.predict(
                prepared_eval[origin_features]
            )
        else:
            transformed_prediction, _ = recursive_tree_prediction(
                fitted,
                prepared_fit,
                prepared_eval,
                origin_features,
                feature_mode=feature_mode,
            )
        predictions.extend(
            inverse_tree_prediction(
                transformed_prediction,
                origin_eval,
                transformer,
            ).tolist()
        )
        forecast_time += time.perf_counter() - forecast_start
    prediction = np.asarray(predictions, dtype=float)
    results = eval_df[["date", "y"]].copy()
    results[forecast_name] = prediction
    metric = {
        "Model": forecast_name,
        "Base Model": model_name,
        "Target Transformation": target_transformation,
        "Model Family": "ML",
        "Feature Eligibility": TREE_FEATURE_ELIGIBILITY,
        **_metrics(eval_df["y"].to_numpy(), prediction),
        "Tuning Time": float(config.get("tuning_seconds", 0.0)),
        "Training Time": training_time,
        "Forecast Time": forecast_time,
        "Best Iteration": best_iteration,
        "Fits": len(eval_df) if recalibrate_each_origin else 1,
        "Order Selections": 0,
        "Order Selection Time": 0.0,
        "Selected Order": None,
        "Selected Seasonal Order": None,
        "Selected Season Length": None,
        "Time Elapsed": (
            float(config.get("tuning_seconds", 0.0))
            + training_time
            + forecast_time
        ),
    }
    return results, metric, fitted


def rolling_tree_shap(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: List[str],
    config: Dict,
    n_jobs: int = CPU_THREADS,
    weight_multipliers: Optional[Dict[str, float]] = None,
    background_size: int = 100,
):
    """Explain each rolling one-step tree forecast on its fitted target scale."""
    import shap

    target_transformation = config.get(
        "target_transformation", "AutoStationary"
    )
    model_name = config["model"]
    rows = []
    for origin in range(len(eval_df)):
        origin_fit = pd.concat(
            [fit_df, eval_df.iloc[:origin]], ignore_index=True
        )
        origin_eval = eval_df.iloc[[origin]]
        origin_features = active_features(origin_fit, feature_columns)
        prepared_fit, prepared_eval, transformer = prepare_tree_origin(
            origin_fit,
            origin_eval,
            origin_features,
            target_transformation=target_transformation,
            freq=config.get("freq", "MS"),
        )
        estimator = _tree_estimator(
            model_name,
            config["parameters"],
            seed=int(config.get("seed", SEED)),
            n_jobs=n_jobs,
        )
        weights = quality_weights(prepared_fit, weight_multipliers)
        fitted, _ = _fit_tree(
            estimator,
            prepared_fit[origin_features],
            prepared_fit["y"],
            weights,
        )
        if model_name == "XGBRandomForest":
            # SHAP releases predating XGBoost's vector-valued base_score
            # metadata cannot parse recent boosters (for example
            # ``"[5.229768E0]"``). XGBoost exposes the same exact
            # path-dependent TreeSHAP algorithm through ``pred_contribs`` and
            # does not rely on SHAP's model-file parser.
            import xgboost as xgb

            evaluation_matrix = xgb.DMatrix(
                prepared_eval[origin_features],
                feature_names=list(origin_features),
                missing=np.nan,
            )
            contributions = np.asarray(
                fitted.get_booster().predict(
                    evaluation_matrix,
                    pred_contribs=True,
                ),
                dtype=float,
            ).reshape(-1, len(origin_features) + 1)[0]
            values = contributions[:-1]
            base_value = float(contributions[-1])
            attribution_method = "XGBoost native TreeSHAP"
            n_background = 0
            explainer = explanation = None
        else:
            n_background = min(int(background_size), len(prepared_fit))
            positions = np.linspace(
                0, len(prepared_fit) - 1, n_background, dtype=int
            )
            background = prepared_fit.iloc[positions][origin_features]
            explainer = shap.TreeExplainer(
                fitted,
                data=background,
                feature_perturbation="interventional",
                model_output="raw",
            )
            explanation = explainer(
                prepared_eval[origin_features],
                check_additivity=False,
            )
            values = np.asarray(explanation.values, dtype=float).reshape(
                -1, len(origin_features)
            )[0]
            base_value = float(
                np.asarray(explanation.base_values, dtype=float).reshape(-1)[0]
            )
            attribution_method = "Interventional TreeSHAP"
        transformed_forecast = float(
            fitted.predict(prepared_eval[origin_features])[0]
        )
        original_forecast = float(
            inverse_tree_prediction(
                np.asarray([transformed_forecast]),
                origin_eval,
                transformer,
            )[0]
        )
        actual = float(origin_eval["y"].iloc[0])
        feature_values = prepared_eval[origin_features].iloc[0]
        for feature, attribution in zip(origin_features, values):
            rows.append(
                {
                    "Origin": pd.Timestamp(origin_eval["date"].iloc[0]),
                    "Model": model_name,
                    "Target Transformation": target_transformation,
                    "Feature": feature,
                    "Variable Group": economic_feature_group(feature),
                    "Feature Value": float(feature_values[feature]),
                    "SHAP": float(attribution),
                    "Absolute SHAP": abs(float(attribution)),
                    "Baseline Forecast": base_value,
                    "Transformed Forecast": transformed_forecast,
                    "Original-Scale Forecast": original_forecast,
                    "Actual": actual,
                    "Background Rows": n_background,
                    "Attribution Method": attribution_method,
                    "Reconstruction Error": float(
                        base_value + values.sum() - transformed_forecast
                    ),
                }
            )
        del fitted, explainer, explanation
    return pd.DataFrame(rows)


def _neuralforecast_frame(
    frame: pd.DataFrame,
    feature_columns: List[str],
    include_target: bool = True,
) -> pd.DataFrame:
    columns = ["date"] + (["y"] if include_target else []) + feature_columns
    data = frame[columns].copy()
    data.insert(0, "unique_id", "inflation")
    return data.rename(columns={"date": "ds"})


def rolling_dl_shaptime(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: List[str],
    model_name: str,
    resolved_config: Dict,
    feature_mode: str = "strict_lagged",
    n_super_times: int = 4,
    seasonal_background_lag: int = 12,
    expected_predictions: Optional[Iterable[float]] = None,
    prediction_atol: float = 1e-4,
    prediction_rtol: float = 1e-4,
    require_prediction_match: bool = True,
    show_progress: bool = False,
):
    """Explain the rolling h=1 fixed-weight neural forecast path.

    The selected model is reconstructed from its frozen configuration and fitted
    once at the external-block origin. At each of the twelve rolling origins,
    exact group Shapley values are computed over contiguous portions of the
    autoregressive target window. The observed history is then updated, while
    model weights and the origin's one-row future-exogenous input remain fixed.
    """
    from neuralforecast import NeuralForecast

    if feature_mode != "strict_lagged":
        raise ValueError(
            "Rolling neural ShapTime requires strict_lagged features"
        )
    if len(eval_df) != 12:
        raise ValueError("Rolling neural ShapTime requires 12 origins")
    fit_df = fit_df.sort_values("date").reset_index(drop=True)
    eval_df = eval_df.sort_values("date").reset_index(drop=True)
    frozen_features = active_features(fit_df, feature_columns)
    non_lagged_features = [
        column
        for column in frozen_features
        if "lag" not in column and column != "q_cpi_reconstruction_period"
    ]
    if non_lagged_features:
        raise ValueError(
            "Neural future regressors are not origin-safe strict lags: "
            f"{non_lagged_features}"
        )
    if pd.concat([fit_df, eval_df], ignore_index=True)[
        frozen_features
    ].isna().any().any():
        raise ValueError("NeuralForecast exogenous tensors cannot contain NaNs")

    input_size = int(resolved_config["input_size"])
    n_super_times = int(n_super_times)
    seasonal_background_lag = int(seasonal_background_lag)
    if n_super_times < 2 or n_super_times > input_size:
        raise ValueError("n_super_times must be between 2 and input_size")
    if seasonal_background_lag < 1:
        raise ValueError("seasonal_background_lag must be positive")
    relative_groups = [
        np.asarray(group, dtype=int)
        for group in np.array_split(np.arange(input_size), n_super_times)
    ]
    all_players = tuple(range(n_super_times))
    coalitions = [
        frozenset(combination)
        for size in range(n_super_times + 1)
        for combination in itertools.combinations(all_players, size)
    ]
    if len(fit_df) < input_size + seasonal_background_lag:
        raise ValueError(
            "Insufficient history for the requested ShapTime background"
        )
    model = _fixed_exogenous_dl_model(
        model_name,
        resolved_config,
        frozen_features,
        forecast_horizon=1,
    )
    nf = NeuralForecast(models=[model], freq="MS")
    try:
        with quiet_neuralforecast_output():
            nf.fit(
                df=_neuralforecast_frame(fit_df, frozen_features),
                val_size=0,
                id_col="unique_id",
                time_col="ds",
                target_col="y",
                verbose=False,
            )
        fit_calls = 1
        history = fit_df.copy()
        rows = []
        reconstructed_predictions = []
        for origin in range(len(eval_df)):
            target_row = eval_df.iloc[[origin]].copy()
            information_cutoff = pd.Timestamp(history["date"].max())
            target_date = pd.Timestamp(target_row["date"].iloc[0])
            if target_date != information_cutoff + pd.offsets.MonthBegin(1):
                raise ValueError(
                    "ShapTime origin is not the next month after its history"
                )
            future = _neuralforecast_frame(
                target_row,
                frozen_features,
                include_target=False,
            )
            if len(future) != 1 or "y" in future.columns:
                raise ValueError(
                    "Each ShapTime origin requires exactly one target-free "
                    "future row"
                )
            current_positions = np.arange(
                len(history) - input_size,
                len(history),
            )
            background_positions = (
                current_positions - seasonal_background_lag
            )
            coalition_values = {}
            for coalition in coalitions:
                hybrid_history = history.copy()
                for player, relative_positions in enumerate(relative_groups):
                    if player in coalition:
                        continue
                    current = current_positions[relative_positions]
                    background = background_positions[relative_positions]
                    hybrid_history.loc[current, "y"] = history.loc[
                        background, "y"
                    ].to_numpy()
                with quiet_neuralforecast_output():
                    prediction = nf.predict(
                        df=_neuralforecast_frame(
                            hybrid_history, frozen_features
                        ),
                        futr_df=future,
                        verbose=False,
                    )
                prediction = prediction.sort_values("ds").reset_index(drop=True)
                if len(prediction) != 1:
                    raise ValueError(
                        "Expected one ShapTime forecast per coalition and origin"
                    )
                coalition_values[coalition] = float(
                    prediction[model_name].iloc[0]
                )

            horizon_values = []
            for player in all_players:
                contribution = 0.0
                for coalition in coalitions:
                    if player in coalition:
                        continue
                    size = len(coalition)
                    weight = (
                        math.factorial(size)
                        * math.factorial(n_super_times - size - 1)
                        / math.factorial(n_super_times)
                    )
                    contribution += weight * (
                        coalition_values[coalition | {player}]
                        - coalition_values[coalition]
                    )
                horizon_values.append(contribution)
            horizon_values = np.asarray(horizon_values, dtype=float)
            empty_forecast = coalition_values[frozenset()]
            full_forecast = coalition_values[frozenset(all_players)]
            reconstructed_predictions.append(full_forecast)
            for player, (relative_positions, attribution) in enumerate(
                zip(relative_groups, horizon_values), start=1
            ):
                oldest_lag = input_size - int(relative_positions[0])
                newest_lag = input_size - int(relative_positions[-1])
                rows.append(
                    {
                        "Origin": target_date,
                        "Information Cutoff": information_cutoff,
                        "Horizon": 1,
                        "Model": model_name,
                        "Super Time": player,
                        "Window": f"t-{oldest_lag}:t-{newest_lag}",
                        "Oldest Lag": oldest_lag,
                        "Newest Lag": newest_lag,
                        "ShapTime": float(attribution),
                        "Absolute ShapTime": abs(float(attribution)),
                        "Background Forecast": float(empty_forecast),
                        "Forecast": float(full_forecast),
                        "Actual": float(target_row["y"].iloc[0]),
                        "Input Size": input_size,
                        "Seasonal Background Lag": seasonal_background_lag,
                        "Reconstruction Error": float(
                            empty_forecast
                            + horizon_values.sum()
                            - full_forecast
                        ),
                        "Fits": fit_calls,
                        "Future Rows": 1,
                        "DL Protocol": DL_PROTOCOL,
                        "ShapTime Protocol": (
                            "rolling_h1_fixed_weights_refit_shaptime_v3"
                        ),
                    }
                )
            # Reveal the target only after every coalition forecast for this
            # origin has been stored.
            history = pd.concat([history, target_row], ignore_index=True)
            if show_progress:
                print(
                    f"ShapTime origin {origin + 1}/{len(eval_df)} complete",
                    flush=True,
                )

        if fit_calls != 1:
            raise AssertionError("ShapTime refitted within the rolling block")
        reconstructed_predictions = np.asarray(
            reconstructed_predictions, dtype=float
        )
        if expected_predictions is not None:
            expected = np.asarray(list(expected_predictions), dtype=float)
            if expected.shape != reconstructed_predictions.shape:
                raise ValueError(
                    "Stored prediction path has an incompatible shape"
                )
            differences = np.abs(expected - reconstructed_predictions)
            prediction_match = bool(np.allclose(
                expected,
                reconstructed_predictions,
                atol=float(prediction_atol),
                rtol=float(prediction_rtol),
            ))
            if require_prediction_match and not prediction_match:
                raise ValueError(
                    "The reconstructed fixed-weight model does not reproduce "
                    "the stored forecast path; maximum absolute difference is "
                    f"{differences.max():.6g}. Attribution was not accepted."
                )
            for row in rows:
                origin_index = int(
                    np.where(
                        eval_df["date"].to_numpy()
                        == np.datetime64(row["Origin"])
                    )[0][0]
                )
                row["Stored Forecast"] = float(expected[origin_index])
                row["Refit Minus Stored Forecast"] = float(
                    reconstructed_predictions[origin_index]
                    - expected[origin_index]
                )
                row["Stored Forecast Max Abs Difference"] = float(
                    differences.max()
                )
                row["Stored Forecast Match"] = prediction_match
                row["Attribution Basis"] = (
                    "same-configuration seed-67 refit; original weights "
                    "were not persisted"
                )
        return pd.DataFrame(rows)
    finally:
        del nf, model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def run_rolling_dl_shaptime_subprocess(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: List[str],
    model_name: str,
    resolved_config: Dict,
    expected_predictions: Iterable[float],
    output_path: Path,
    feature_mode: str = "strict_lagged",
    n_super_times: int = 4,
    seasonal_background_lag: int = 12,
    require_prediction_match: bool = False,
) -> pd.DataFrame:
    """Run neural ShapTime outside a SHAP/LightGBM notebook process.

    PyTorch and conda-forge SHAP can load incompatible Intel and LLVM OpenMP
    runtimes on Windows. A clean worker process keeps the neural stack isolated
    and writes the checkpoint atomically.
    """
    output_path = Path(output_path).resolve()
    worker = Path(__file__).with_name("exogenous_shaptime_worker.py").resolve()
    if not worker.exists():
        raise FileNotFoundError(f"ShapTime worker is missing: {worker}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="exogenous_shaptime_") as temp:
        temp_dir = Path(temp)
        fit_path = temp_dir / "fit.csv"
        eval_path = temp_dir / "evaluation.csv"
        request_path = temp_dir / "request.json"
        fit_df.to_csv(fit_path, index=False)
        eval_df.to_csv(eval_path, index=False)
        request = {
            "fit_path": str(fit_path),
            "eval_path": str(eval_path),
            "feature_columns": list(feature_columns),
            "model_name": model_name,
            "resolved_config": copy.deepcopy(resolved_config),
            "expected_predictions": [
                float(value) for value in expected_predictions
            ],
            "output_path": str(output_path),
            "feature_mode": feature_mode,
            "n_super_times": int(n_super_times),
            "seasonal_background_lag": int(seasonal_background_lag),
            "require_prediction_match": bool(require_prediction_match),
        }
        request_path.write_text(
            json.dumps(request, indent=2, default=str), encoding="utf-8"
        )
        environment = os.environ.copy()
        environment.pop("KMP_DUPLICATE_LIB_OK", None)
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        process = subprocess.Popen(
            [
                sys.executable,
                str(worker),
                "--request",
                str(request_path),
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        worker_output = []
        assert process.stdout is not None
        for line in process.stdout:
            worker_output.append(line)
            print(f"[ShapTime worker] {line}", end="")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                "The isolated ShapTime worker failed with exit code "
                f"{return_code}:\n{''.join(worker_output)}"
            )
    if not output_path.exists():
        raise RuntimeError("ShapTime worker completed without an output file")
    return pd.read_csv(output_path, parse_dates=["Origin", "Information Cutoff"])


# Backward import guard: callers receive the corrected rolling implementation.
block_dl_shaptime = rolling_dl_shaptime


def evaluate_autoarimax(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: List[str],
    recalibrate_each_origin: bool = True,
):
    """Evaluate AutoARIMAX with one order search and fixed-order recalibration."""
    from statsforecast import StatsForecast
    from statsforecast.models import ARIMA, AutoARIMA

    if pd.concat([fit_df, eval_df], ignore_index=True)[
        feature_columns
    ].isna().any().any():
        raise ValueError("AutoARIMAX requires complete exogenous regressors")

    def autoarima():
        return AutoARIMA(
            season_length=12,
            max_p=8,
            max_q=8,
            max_P=3,
            max_Q=3,
            max_order=15,
            max_d=2,
            max_D=1,
            stepwise=True,
            nmodels=200,
            approximation=False,
            alias="AutoARIMAX",
        )

    if recalibrate_each_origin:
        fit_y = fit_df["y"].astype(float).to_numpy()
        fit_X = fit_df[feature_columns].astype(float).to_numpy()
        automatic = autoarima()
        selection_start = time.perf_counter()
        automatic.fit(y=fit_y, X=fit_X)
        order_selection_time = time.perf_counter() - selection_start
        arma = automatic.model_["arma"]
        coefficients = automatic.model_["coef"]
        selected_order = tuple(int(arma[index]) for index in (0, 5, 1))
        selected_seasonal_order = tuple(
            int(arma[index]) for index in (2, 6, 3)
        )
        selected_season_length = int(arma[4])
        include_mean = "intercept" in coefficients
        include_drift = "drift" in coefficients
        predictions = []
        refit_time = 0.0
        forecast_time = 0.0
        for origin in range(len(eval_df)):
            origin_fit = pd.concat(
                [fit_df, eval_df.iloc[:origin]],
                ignore_index=True,
            )
            fixed_order = ARIMA(
                order=selected_order,
                season_length=selected_season_length,
                seasonal_order=selected_seasonal_order,
                include_mean=include_mean,
                include_drift=include_drift,
                method=automatic.method or "CSS-ML",
                alias="AutoARIMAX",
            )
            refit_start = time.perf_counter()
            fixed_order.fit(
                y=origin_fit["y"].astype(float).to_numpy(),
                X=origin_fit[feature_columns].astype(float).to_numpy(),
            )
            refit_time += time.perf_counter() - refit_start
            forecast_start = time.perf_counter()
            forecast = fixed_order.predict(
                h=1,
                X=eval_df.iloc[[origin]][feature_columns]
                .astype(float)
                .to_numpy(),
            )
            forecast_time += time.perf_counter() - forecast_start
            predictions.append(float(forecast["mean"][0]))
        prediction = np.asarray(predictions, dtype=float)
        actual = eval_df["y"].to_numpy()
        dates = eval_df["date"].to_numpy()
        training_time = order_selection_time + refit_time
    else:
        data = fit_df[["date", "y"] + feature_columns].copy()
        data.insert(0, "unique_id", "inflation")
        data = data.rename(columns={"date": "ds"})
        future = eval_df[["date"] + feature_columns].copy()
        future.insert(0, "unique_id", "inflation")
        future = future.rename(columns={"date": "ds"})
        sf = StatsForecast(models=[autoarima()], freq="MS", n_jobs=CPU_THREADS)
        start = time.perf_counter()
        sf.fit(
            df=data,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
        )
        training_time = time.perf_counter() - start
        order_selection_time = training_time
        selected_order = None
        selected_seasonal_order = None
        selected_season_length = 12
        forecast_start = time.perf_counter()
        forecast = sf.predict(h=len(eval_df), X_df=future)
        forecast_time = time.perf_counter() - forecast_start
        forecast = forecast.sort_values("ds").reset_index(drop=True)
        prediction = forecast["AutoARIMAX"].astype(float).to_numpy()
        actual = eval_df["y"].astype(float).to_numpy()
        dates = eval_df["date"].to_numpy()
    elapsed = training_time + forecast_time
    results = pd.DataFrame(
        {"date": dates, "y": actual, "AutoARIMAX": prediction}
    )
    metric = {
        "Model": "AutoARIMAX",
        "Base Model": "AutoARIMAX",
        "Target Transformation": "Original",
        "Model Family": "Classical",
        **_metrics(actual, prediction),
        "Tuning Time": 0.0,
        "Training Time": training_time,
        "Forecast Time": forecast_time,
        "Best Iteration": None,
        "Fits": len(eval_df) if recalibrate_each_origin else 1,
        "Order Selections": 1,
        "Order Selection Time": order_selection_time,
        "Selected Order": selected_order,
        "Selected Seasonal Order": selected_seasonal_order,
        "Selected Season Length": selected_season_length,
        "Time Elapsed": elapsed,
    }
    return results, metric


def load_exogenous_dl_search_configs(
    feature_columns: List[str],
    config_dir: str = "thesis/model_configs/dl",
    input_size_candidates: Optional[List[int]] = None,
    num_samples: int = 4,
    startup_trials: int = 2,
    active_models: Optional[Iterable[str]] = None,
    forecast_horizon: int = 1,
) -> Dict[str, Dict]:
    """Adapt the existing architecture spaces to exogenous models."""
    input_size_candidates = input_size_candidates or [12, 24, 36, 48]
    active_models = tuple(active_models or PRIMARY_EXOGENOUS_DL_MODELS)
    unknown = set(active_models).difference(DL_CONFIG_FILES)
    if unknown:
        raise ValueError(f"Unsupported exogenous DL models: {sorted(unknown)}")
    configs = {}
    for model_name in active_models:
        filename = DL_CONFIG_FILES[model_name]
        with (Path(config_dir) / filename).open(encoding="utf-8") as file:
            config = json.load(file)
        config["_source"] = (
            config.get("_source", model_name)
            + "; adapted to Forte exogenous regressors"
        )
        config["_input_size_candidates"] = list(input_size_candidates)
        config["_auto"]["num_samples"] = int(num_samples)
        config["_auto"]["n_startup_trials"] = int(startup_trials)
        config["_auto"]["refit_with_val"] = True
        config["futr_exog_list"] = list(feature_columns)
        if model_name == "AutoNBEATSx" and forecast_horizon == 1:
            # NeuralForecast's trend and seasonality bases collapse when h=1.
            # Three generic identity stacks retain the N-BEATSx architecture
            # and its exogenous inputs without imposing an invalid basis.
            config["stack_types"] = ["identity", "identity", "identity"]
            config["_adaptations"] = list(config.get("_adaptations", [])) + [
                "h=1 generic identity stacks"
            ]
        config["logger"] = False
        config["enable_progress_bar"] = False
        config["enable_model_summary"] = False
        config.update(neuralforecast_compute_parameters())
        configs[model_name] = config
    return configs


def _build_auto_exogenous_model(
    model_name: str,
    config: Dict,
    forecast_horizon: int = 1,
):
    import optuna
    from neuralforecast.auto import (
        AutoAutoformer,
        AutoGRU,
        AutoKAN,
        AutoLSTM,
        AutoNBEATSx,
        AutoNHITS,
        AutoRNN,
        AutoTFT,
        AutoTiDE,
        AutoTSMixerx,
        AutoTimesNet,
        AutoVanillaTransformer,
    )
    from neuralforecast.losses.pytorch import MAE

    from dl_evaluation import (
        _cleanup_optuna_trial,
        _optuna_config,
        ensure_timesnet_fft_safe,
        model_parameters,
    )

    classes = {
        "AutoTFT": AutoTFT,
        "AutoNBEATSx": AutoNBEATSx,
        "AutoTimesNet": AutoTimesNet,
        "AutoKAN": AutoKAN,
        "AutoTiDE": AutoTiDE,
        "AutoTSMixerx": AutoTSMixerx,
        "AutoRNN": AutoRNN,
        "AutoLSTM": AutoLSTM,
        "AutoGRU": AutoGRU,
        "AutoVanillaTransformer": AutoVanillaTransformer,
        "AutoAutoformer": AutoAutoformer,
        "AutoNHITS": AutoNHITS,
    }
    auto = config["_auto"]
    auto_kwargs = {}
    model_config = config.copy()
    if model_name == "AutoTimesNet":
        model_config = ensure_timesnet_fft_safe(
            model_config, int(forecast_horizon)
        )
    if model_name == "AutoTSMixerx":
        model_config["n_series"] = 1
        auto_kwargs["n_series"] = 1
    return classes[model_name](
        h=int(forecast_horizon),
        loss=MAE(),
        valid_loss=MAE(),
        config=_optuna_config(model_config),
        search_alg=optuna.samplers.TPESampler(
            seed=int(model_parameters(model_config).get("random_seed", SEED)),
            n_startup_trials=int(auto["n_startup_trials"]),
        ),
        num_samples=int(auto["num_samples"]),
        refit_with_val=bool(auto.get("refit_with_val", True)),
        verbose=bool(auto.get("verbose", False)),
        backend="optuna",
        callbacks=[_cleanup_optuna_trial],
        alias=model_name,
        **auto_kwargs,
    )


def tune_exogenous_dl_models(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: List[str],
    configs: Dict[str, Dict],
    progress_label: str = "",
    forecast_horizon: int = 1,
):
    """Tune each architecture by 12 rolling h=1 forecasts with one fit/trial."""
    import optuna

    if int(forecast_horizon) != 1:
        raise ValueError("Exogenous neural tuning requires h=1")
    if len(validation_df) != 12:
        raise ValueError("Exogenous neural tuning requires 12 validation origins")

    frozen_features = active_features(train_df, feature_columns)
    if pd.concat([train_df, validation_df])[frozen_features].isna().any().any():
        raise ValueError("NeuralForecast exogenous tensors cannot contain NaNs")
    resolved = {}
    timing_rows = []
    for index, (model_name, config) in enumerate(configs.items(), start=1):
        prefix = f"{progress_label} | " if progress_label else ""
        print(f"{prefix}DL tuning {index}/{len(configs)}: {model_name}")
        from dl_evaluation import _optuna_config, ensure_timesnet_fft_safe

        search_config = copy.deepcopy(config)
        if model_name == "AutoTimesNet":
            search_config = ensure_timesnet_fft_safe(search_config, 1)
        configure = _optuna_config(search_config)
        auto = search_config["_auto"]
        trial_records = {}

        def objective(trial):
            trial_config = configure(trial)
            candidate_name = f"{model_name}__trial_{trial.number}"
            results, metric = evaluate_fixed_exogenous_dl(
                train_df,
                validation_df,
                frozen_features,
                model_name,
                trial_config,
                tuning_time=0.0,
                recalibrate_each_origin=False,
                output_name=candidate_name,
            )
            trial_records[int(trial.number)] = {
                "candidate_id": candidate_name,
                "architecture": model_name,
                "trial_number": int(trial.number),
                "resolved_config": trial_config,
                "validation_mae": float(metric["mae"]),
                "results": {
                    "date": [
                        pd.Timestamp(value).isoformat()
                        for value in results["date"]
                    ],
                    "y": results["y"].astype(float).tolist(),
                    "prediction": results[candidate_name].astype(float).tolist(),
                },
                "metric": metric,
            }
            return float(metric["mae"])

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=SEED,
                n_startup_trials=int(auto["n_startup_trials"]),
            ),
        )
        start = time.perf_counter()
        study.optimize(
            objective,
            n_trials=int(auto["num_samples"]),
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - start
        completed = [
            trial_records[trial.number]
            for trial in study.trials
            if trial.number in trial_records
        ]
        if len(completed) != int(auto["num_samples"]):
            raise RuntimeError(
                f"{model_name}: expected {auto['num_samples']} completed "
                f"configurations, found {len(completed)}"
            )
        best = copy.deepcopy(trial_records[study.best_trial.number])
        best_config = copy.deepcopy(best["resolved_config"])
        best_config["futr_exog_list"] = list(frozen_features)
        best_config["_candidate_id"] = best["candidate_id"]
        best_config["_trial_number"] = best["trial_number"]
        best_config["_validation_mae"] = best["validation_mae"]
        best_config["_dl_protocol"] = DL_PROTOCOL
        best_config["_trial_records"] = completed
        resolved[model_name] = best_config
        timing_rows.append(
            {
                "Model": model_name,
                "Tuning Time": elapsed,
                "Auto Refit Time": 0.0,
                "Time Elapsed": elapsed,
                "Best Validation Loss": float(study.best_value),
                "Completed Configurations": len(completed),
            }
        )
    return resolved, pd.DataFrame(timing_rows)


def stored_dl_trial_evaluation(
    record: Dict,
    tuning_time: float = 0.0,
):
    """Restore the external-validation forecast made inside one Optuna trial."""
    candidate_id = record["candidate_id"]
    saved = record["results"]
    results = pd.DataFrame(
        {
            "date": pd.to_datetime(saved["date"]),
            "y": np.asarray(saved["y"], dtype=float),
            candidate_id: np.asarray(saved["prediction"], dtype=float),
        }
    )
    if len(results) != 12 or results["date"].nunique() != 12:
        raise ValueError("Stored neural trial does not contain 12 origins")
    metric = copy.deepcopy(record["metric"])
    metric["Model"] = candidate_id
    metric["Base Model"] = record["architecture"]
    metric["Tuning Time"] = float(tuning_time)
    # The architecture search time already contains every trial fit and
    # forecast, including this selected trial.
    metric["Time Elapsed"] = float(tuning_time)
    return results, metric


def _fixed_exogenous_dl_model(
    model_name: str,
    resolved_config: Dict,
    feature_columns: List[str],
    forecast_horizon: int = 1,
    alias: Optional[str] = None,
):
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import (
        Autoformer,
        GRU,
        KAN,
        LSTM,
        NBEATSx,
        NHITS,
        RNN,
        TFT,
        TSMixerx,
        TiDE,
        TimesNet,
        VanillaTransformer,
    )

    classes = {
        "AutoTFT": TFT,
        "AutoNBEATSx": NBEATSx,
        "AutoTimesNet": TimesNet,
        "AutoKAN": KAN,
        "AutoTiDE": TiDE,
        "AutoTSMixerx": TSMixerx,
        "AutoRNN": RNN,
        "AutoLSTM": LSTM,
        "AutoGRU": GRU,
        "AutoVanillaTransformer": VanillaTransformer,
        "AutoAutoformer": Autoformer,
        "AutoNHITS": NHITS,
    }
    parameters = {
        key: value
        for key, value in resolved_config.items()
        if not key.startswith("_")
    }
    for key in ("h", "alias", "loss", "valid_loss", "n_series"):
        parameters.pop(key, None)
    parameters["futr_exog_list"] = list(feature_columns)
    parameters["loss"] = MAE()
    parameters["valid_loss"] = MAE()
    parameters["early_stop_patience_steps"] = -1
    parameters.update(neuralforecast_compute_parameters())
    if model_name == "AutoTSMixerx":
        parameters["n_series"] = 1
    return classes[model_name](
        h=int(forecast_horizon),
        alias=alias or model_name,
        **parameters,
    )


def evaluate_fixed_exogenous_dl(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_columns: List[str],
    model_name: str,
    resolved_config: Dict,
    tuning_time: float = 0.0,
    recalibrate_each_origin: bool = False,
    output_name: Optional[str] = None,
    feature_mode: str = "strict_lagged",
):
    """Fit once, then issue rolling one-step forecasts with frozen weights."""
    from neuralforecast import NeuralForecast

    if recalibrate_each_origin:
        raise ValueError(
            "Exogenous neural models use fixed weights across each block"
        )
    if len(eval_df) != 12:
        raise ValueError("Exogenous neural evaluation requires 12 origins")
    if feature_mode != "strict_lagged":
        raise ValueError(
            "Rolling exogenous neural forecasts require strict_lagged features"
        )

    fit_df = fit_df.sort_values("date").reset_index(drop=True)
    eval_df = eval_df.sort_values("date").reset_index(drop=True)
    frozen_features = active_features(fit_df, feature_columns)
    non_lagged_features = [
        column
        for column in frozen_features
        if "lag" not in column and column != "q_cpi_reconstruction_period"
    ]
    if non_lagged_features:
        raise ValueError(
            "Neural future regressors are not origin-safe strict lags: "
            f"{non_lagged_features}"
        )
    combined = pd.concat([fit_df, eval_df], ignore_index=True)
    if combined[frozen_features].isna().any().any():
        raise ValueError("NeuralForecast exogenous tensors cannot contain NaNs")
    forecast_name = output_name or model_name
    model = None
    nf = None
    try:
        model = _fixed_exogenous_dl_model(
            model_name,
            resolved_config,
            frozen_features,
            forecast_horizon=1,
            alias=forecast_name,
        )
        nf = NeuralForecast(models=[model], freq="MS")
        fit_data = _neuralforecast_frame(fit_df, frozen_features)
        start = time.perf_counter()
        fit_calls = 0
        with quiet_neuralforecast_output():
            nf.fit(
                df=fit_data,
                val_size=0,
                id_col="unique_id",
                time_col="ds",
                target_col="y",
                verbose=False,
            )
        fit_calls += 1
        training_time = time.perf_counter() - start

        history = fit_df.copy().sort_values("date").reset_index(drop=True)
        prediction_values = []
        prediction_dates = []
        forecast_time = 0.0
        for origin in range(len(eval_df)):
            future_row = eval_df.iloc[[origin]].copy()
            information_cutoff = pd.Timestamp(history["date"].max())
            target_date = pd.Timestamp(future_row["date"].iloc[0])
            assert target_date > information_cutoff
            expected_target_date = information_cutoff + pd.offsets.MonthBegin(1)
            assert target_date == expected_target_date
            assert not any("current" in column for column in frozen_features)
            future = _neuralforecast_frame(
                future_row,
                frozen_features,
                include_target=False,
            )
            assert len(future) == 1
            assert "y" not in future.columns
            assert pd.Timestamp(future["ds"].iloc[0]) == target_date
            forecast_start = time.perf_counter()
            with quiet_neuralforecast_output():
                forecast = nf.predict(
                    df=_neuralforecast_frame(history, frozen_features),
                    futr_df=future,
                    verbose=False,
                )
            forecast_time += time.perf_counter() - forecast_start
            forecast = forecast.sort_values("ds").reset_index(drop=True)
            if len(forecast) != 1:
                raise ValueError("Expected exactly one neural forecast per origin")
            prediction_values.append(float(forecast[forecast_name].iloc[0]))
            prediction_dates.append(pd.Timestamp(forecast["ds"].iloc[0]))
            # Reveal the target only after its forecast has been stored.
            history = pd.concat([history, future_row], ignore_index=True)

        prediction = np.asarray(prediction_values, dtype=float)
        actual = eval_df["y"].astype(float).to_numpy()
        if len(set(prediction_dates)) != 12:
            raise ValueError("Neural rolling evaluation did not produce 12 dates")
        assert fit_calls == 1
        results = pd.DataFrame(
            {"date": prediction_dates, "y": actual, forecast_name: prediction}
        )
        metric = {
            "Model": forecast_name,
            "Base Model": model_name,
            "Target Transformation": "Original",
            "Model Family": "DL",
            "DL Protocol": DL_PROTOCOL,
            **_metrics(actual, prediction),
            "Tuning Time": tuning_time,
            "Training Time": training_time,
            "Forecast Time": forecast_time,
            "Best Iteration": None,
            "Fits": 1,
            "Order Selections": 0,
            "Order Selection Time": 0.0,
            "Selected Order": None,
            "Selected Seasonal Order": None,
            "Selected Season Length": None,
            "Time Elapsed": tuning_time + training_time + forecast_time,
        }
        return results, metric
    finally:
        del nf, model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
