"""Fair grid, random, and TPE search comparison for one recursive ML model."""

from __future__ import annotations

import json
import time
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import ParameterGrid
from thesis.runtime import CPU_THREADS

try:
    from thesis.ml_evaluation import (
        evaluate_recursive_ml_models,
        expanding_window_splits,
    )
except ModuleNotFoundError:  # Support execution from ``thesis/``.
    from ml_evaluation import evaluate_recursive_ml_models, expanding_window_splits


SEARCH_BOUNDS = {
    "learning_rate": (0.005, 0.10),
    "num_leaves": (8, 96),
    "min_child_samples": (5, 60),
}


def lightgbm_estimator(params: Mapping, seed: int = 67) -> LGBMRegressor:
    """Fixed LightGBM specification; only the declared search dimensions vary."""
    return LGBMRegressor(
        objective="regression",
        n_estimators=400,
        max_depth=-1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=int(seed),
        deterministic=True,
        force_col_wise=True,
        n_jobs=CPU_THREADS,
        verbosity=-1,
        **dict(params),
    )


def comparison_folds(
    ts_train: pd.DataFrame,
    *,
    lags: Iterable[int],
    horizon: int = 12,
    n_splits: int = 3,
    time_col: str = "ds",
):
    """Build the same expanding Rep-Holdout-O folds for every search method."""
    ordered_lags = sorted(set(int(lag) for lag in lags))
    return expanding_window_splits(
        ts_train,
        horizon=int(horizon),
        n_splits=int(n_splits),
        time_col=time_col,
        min_train_periods=max(ordered_lags) + 1,
    )


def evaluate_configuration(
    folds,
    params: Mapping,
    *,
    lags: Iterable[int],
    seed: int = 67,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    freq: str | None = None,
) -> tuple[float, float]:
    """Return mean fold RMSE and wall-clock seconds for one configuration."""
    estimator = lightgbm_estimator(params, seed=seed)
    scores = []
    started = time.perf_counter()
    for fit_fold, validation_fold in folds:
        _, metrics, _, _ = evaluate_recursive_ml_models(
            fit_fold,
            validation_fold,
            {"LightGBM": estimator},
            list(lags),
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            freq=freq,
            target_transformation="Original",
        )
        scores.extend(metrics["rmse"].astype(float).tolist())
    elapsed = time.perf_counter() - started
    if not scores:
        raise ValueError("No fold scores were produced")
    return float(np.mean(scores)), float(elapsed)


def grid_configurations() -> list[dict]:
    """A 4 x 4 x 4 lattice spanning the common bounded domain."""
    return list(ParameterGrid({
        "learning_rate": np.geomspace(
            SEARCH_BOUNDS["learning_rate"][0],
            SEARCH_BOUNDS["learning_rate"][1],
            4,
        ).tolist(),
        "num_leaves": [8, 37, 67, 96],
        "min_child_samples": [5, 23, 42, 60],
    }))


def random_configurations(n_trials: int, seed: int = 67) -> list[dict]:
    """Independent draws from the same bounded domain as TPE."""
    rng = np.random.default_rng(seed)
    configurations = []
    for _ in range(int(n_trials)):
        configurations.append({
            "learning_rate": float(np.exp(rng.uniform(
                np.log(SEARCH_BOUNDS["learning_rate"][0]),
                np.log(SEARCH_BOUNDS["learning_rate"][1]),
            ))),
            "num_leaves": int(rng.integers(
                SEARCH_BOUNDS["num_leaves"][0],
                SEARCH_BOUNDS["num_leaves"][1] + 1,
            )),
            "min_child_samples": int(rng.integers(
                SEARCH_BOUNDS["min_child_samples"][0],
                SEARCH_BOUNDS["min_child_samples"][1] + 1,
            )),
        })
    return configurations


def _trial_frame(method: str, evaluations: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(evaluations)
    frame.insert(0, "Method", method)
    frame.insert(1, "Trial", np.arange(1, len(frame) + 1))
    frame["Best RMSE"] = frame["RMSE"].cummin()
    frame["Cumulative Time"] = frame["Trial Time"].cumsum()
    return frame[
        [
            "Method", "Trial", "RMSE", "Best RMSE",
            "Trial Time", "Cumulative Time", "Parameters",
        ]
    ]


def run_configuration_sequence(
    method: str,
    configurations: Iterable[Mapping],
    folds,
    *,
    lags: Iterable[int],
    seed: int = 67,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    freq: str | None = None,
) -> pd.DataFrame:
    """Evaluate an ordered grid or random configuration sequence."""
    evaluations = []
    for trial_number, params in enumerate(configurations, start=1):
        score, elapsed = evaluate_configuration(
            folds,
            params,
            lags=lags,
            seed=seed,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            freq=freq,
        )
        evaluations.append({
            "RMSE": score,
            "Trial Time": elapsed,
            "Parameters": json.dumps(dict(params), sort_keys=True),
        })
        print(
            f"{method} trial {trial_number}: "
            f"RMSE={score:.6f}, {elapsed:.2f}s"
        )
    return _trial_frame(method, evaluations)


def run_tpe_search(
    n_trials: int,
    folds,
    *,
    lags: Iterable[int],
    seed: int = 67,
    startup_trials: int = 5,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    freq: str | None = None,
) -> pd.DataFrame:
    """Sequential Optuna/TPE search with per-trial timing."""
    import optuna

    sampler = optuna.samplers.TPESampler(
        seed=int(seed),
        n_startup_trials=min(int(startup_trials), int(n_trials)),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    evaluations = []
    for trial_number in range(1, int(n_trials) + 1):
        trial = study.ask()
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate", *SEARCH_BOUNDS["learning_rate"], log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", *SEARCH_BOUNDS["num_leaves"]
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", *SEARCH_BOUNDS["min_child_samples"]
            ),
        }
        score, elapsed = evaluate_configuration(
            folds,
            params,
            lags=lags,
            seed=seed,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            freq=freq,
        )
        study.tell(trial, score)
        evaluations.append({
            "RMSE": score,
            "Trial Time": elapsed,
            "Parameters": json.dumps(params, sort_keys=True),
        })
        print(
            f"Bayesian optimization trial {trial_number}: "
            f"RMSE={score:.6f}, {elapsed:.2f}s"
        )
    return _trial_frame("Bayesian optimization", evaluations)


def summarize_searches(
    trials: pd.DataFrame,
    *,
    n_splits: int,
) -> pd.DataFrame:
    """Summarize sample efficiency and total model-fitting effort."""
    rows = []
    for method, group in trials.groupby("Method", sort=False):
        best_index = group["RMSE"].idxmin()
        rows.append({
            "Method": method,
            "Best RMSE": float(group.loc[best_index, "RMSE"]),
            "Best Trial": int(group.loc[best_index, "Trial"]),
            "Trials": int(len(group)),
            "Fold Fits": int(len(group) * n_splits),
            "Time Elapsed": float(group["Trial Time"].sum()),
            "Best Parameters": group.loc[best_index, "Parameters"],
        })
    return pd.DataFrame(rows).sort_values("Best RMSE").reset_index(drop=True)
