"""Utilities for the thesis global synthetic forecasting experiment.

The global files are balanced long-format panels: every row is one observation,
and ``series_id`` identifies one of 100 related series.  Tree models share one
pooled lag-to-target mapping within a partition; NeuralForecast models share one
set of network weights across the complete panel.
"""

from __future__ import annotations

import copy
import importlib
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.transforms.target_transformations import AutoStationaryTransformer


SEED = 67
GLOBAL_HORIZON = 10

GLOBAL_SYNTHETIC_DATASETS = {
    "autoregressive": {
        "label": "Global synthetic AR(1)",
        "file": "thesis/data/synthetic_ts_global_autoregressive.csv",
        "seasonality": 1,
    },
    "pseudo_periodic": {
        "label": "Global synthetic pseudo-periodic",
        "file": "thesis/data/synthetic_ts_global_pseudo_periodic.csv",
        "seasonality": 4,
    },
    "sinusoidal_trend_noise": {
        "label": "Global synthetic sinusoidal trend plus noise",
        "file": "thesis/data/synthetic_ts_global_sinusoidal_trend_noise.csv",
        "seasonality": 4,
    },
}


GLOBAL_DL_CATEGORIES = {
    "AutoRNN": "Recurrent",
    "AutoLSTM": "Recurrent",
    "AutoGRU": "Recurrent",
    "AutoVanillaTransformer": "Attention",
    "AutoAutoformer": "Attention",
    "AutoPatchTST": "Attention",
    "AutoiTransformer": "Attention",
    "AutoTFT": "Attention",
    "AutoTimesNet": "Convolutional",
    "AutoNLinear": "Linear",
    "AutoTSMixer": "MLP",
    "AutoTiDE": "MLP",
    "AutoNBEATS": "Basis expansion",
    "AutoNHITS": "Basis expansion",
    "AutoKAN": "KAN",
}


def _forecast_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Match the thesis percentage-bias definition without eager heavy imports."""
    actual_sum = float(np.sum(actual))
    if actual_sum == 0:
        return 0.0
    return 100.0 * float(np.sum(predicted) - actual_sum) / abs(actual_sum)


def load_global_panel(
    dataset_name: str,
    dataset_configs: Dict[str, Dict] = GLOBAL_SYNTHETIC_DATASETS,
    synthetic_start: str = "2000-01-01",
) -> Tuple[pd.DataFrame, Dict]:
    """Load and validate one balanced global synthetic panel."""
    if dataset_name not in dataset_configs:
        raise ValueError(
            f"Unknown global dataset {dataset_name!r}; "
            f"choose from {sorted(dataset_configs)}"
        )
    config = dict(dataset_configs[dataset_name])
    frame = pd.read_csv(config["file"])
    required = {"series_id", "time", "value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{config['file']} is missing columns {sorted(missing)}")
    if frame.duplicated(["series_id", "time"]).any():
        raise ValueError("Global panel contains duplicate series/time observations")

    frame = frame.sort_values(["series_id", "time"]).reset_index(drop=True)
    sizes = frame.groupby("series_id", sort=False).size()
    if sizes.empty or sizes.nunique() != 1:
        raise ValueError("Every global series must contain the same number of rows")
    if frame.groupby("time")["series_id"].nunique().nunique() != 1:
        raise ValueError("All global series must share aligned generated time steps")

    generated_step = frame["time"].round().astype(int)
    panel = pd.DataFrame(
        {
            "unique_id": frame["series_id"].astype(str),
            "ds": pd.Timestamp(synthetic_start)
            + pd.to_timedelta(generated_step, unit="D"),
            "y": frame["value"].astype(float),
        }
    )
    config.update(
        {
            "freq": "D",
            "horizon": GLOBAL_HORIZON,
            "n_series": int(sizes.size),
            "n_timesteps": int(sizes.iloc[0]),
            "n_rows": int(len(panel)),
        }
    )
    return panel, config


def split_global_panel(
    panel: pd.DataFrame,
    horizon: int = GLOBAL_HORIZON,
) -> Dict[str, pd.DataFrame]:
    """Create aligned 480/10/10-style expanding temporal partitions."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    ordered = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True).copy()
    position = ordered.groupby("unique_id").cumcount()
    size = ordered.groupby("unique_id")["unique_id"].transform("size")
    if int(size.min()) <= 2 * horizon:
        raise ValueError("Each series needs more than two forecast horizons")

    train_end = size - 2 * horizon
    validation_end = size - horizon
    train = ordered.loc[position < train_end].copy()
    validation = ordered.loc[
        position.ge(train_end) & position.lt(validation_end)
    ].copy()
    test = ordered.loc[position.ge(validation_end)].copy()
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def fit_and_evaluation_frames(
    splits: Dict[str, pd.DataFrame],
    evaluation_split: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/validation or train+validation/test without leakage."""
    if evaluation_split == "validation":
        return splits["train"].copy(), splits["validation"].copy()
    if evaluation_split == "test":
        fit = pd.concat(
            [splits["train"], splits["validation"]], ignore_index=True
        )
        return fit, splits["test"].copy()
    raise ValueError("evaluation_split must be 'validation' or 'test'")


def panel_audit(panel: pd.DataFrame, splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compact data-layout summary for the notebook."""
    counts = panel.groupby("unique_id").size()
    row = {
        "Rows": len(panel),
        "Series": panel["unique_id"].nunique(),
        "Time steps per series": int(counts.iloc[0]),
        "Balanced": bool(counts.nunique() == 1),
    }
    for split_name, frame in splits.items():
        row[f"{split_name.title()} steps"] = int(
            frame.groupby("unique_id").size().iloc[0]
        )
    return pd.DataFrame([row])


def _read_json(path: Path) -> Dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def manual_ml_records(
    dataset_name: str,
    selections: Dict[str, Dict[str, Dict[str, str]]],
    candidate_path: str = "thesis/model_configs/ml_validation_candidates.json",
) -> Dict[str, Dict]:
    """Resolve explicit global-ML choices from saved manual validation candidates.

    ``selections`` is deliberately edited by the researcher after reviewing
    the other experiments. Each target entry supplies ``source_dataset`` and
    ``model``; no global test result can alter that choice.
    """
    if dataset_name not in selections:
        raise ValueError(f"No manual global-ML choice for {dataset_name}")
    candidates = _read_json(Path(candidate_path))
    declared = selections[dataset_name]
    unknown_targets = set(declared) - {"Original", "AutoStationary"}
    if unknown_targets:
        raise ValueError(
            f"{dataset_name}: unknown target representations "
            f"{sorted(unknown_targets)}"
        )
    if not declared:
        raise ValueError(f"{dataset_name}: declare at least one ML choice")
    records = {}
    for target in ("Original", "AutoStationary"):
        choice = declared.get(target)
        if not choice:
            continue
        source_dataset = choice["source_dataset"]
        model_name = choice["model"]
        try:
            config = candidates["per_dataset"][source_dataset][target][model_name]
        except KeyError as error:
            raise KeyError(
                f"No saved {source_dataset}/{target}/{model_name} candidate. "
                "Run that manual dataset's ML validation export first."
            ) from error
        records[target] = {
            "base_model": model_name,
            "model": (
                model_name
                if target == "Original"
                else f"{model_name}_AutoStationary"
            ),
            "config": config,
            "source_dataset": source_dataset,
            "selection_note": "Researcher-declared before global evaluation",
        }
    return records


def configure_global_dl_models(
    base_configs: Dict[str, Dict],
    *,
    input_size_candidates: Iterable[int],
    num_samples: int,
    startup_trials: int,
    max_steps: int,
    seed: int = SEED,
) -> Dict[str, Dict]:
    """Apply one common global-panel Auto-search protocol to all 15 models."""
    candidates = [int(value) for value in input_size_candidates]
    if not candidates or min(candidates) < 1:
        raise ValueError("Global DL input-size candidates must be positive")
    if not 0 < int(startup_trials) <= int(num_samples):
        raise ValueError("startup_trials must be between 1 and num_samples")
    configured = copy.deepcopy(base_configs)
    for config in configured.values():
        config["input_size"] = candidates[0]
        config["_input_size_candidates"] = candidates
        config["random_seed"] = int(seed)
        config["max_steps"] = int(max_steps)
        config["_auto"]["num_samples"] = int(num_samples)
        config["_auto"]["n_startup_trials"] = int(startup_trials)
        config.setdefault("enable_progress_bar", False)
        config.setdefault("enable_model_summary", False)
        config.setdefault("logger", False)
    return configured


def global_dl_record(
    model_name: str,
    resolved_parameters: Dict,
    *,
    source: str,
    validation_rmse: float,
) -> Dict:
    """Build the frozen record used for a final global test refit."""
    return {
        "base_model": model_name,
        "model": model_name,
        "config": {
            "class": model_name,
            "module": "neuralforecast.models",
            "parameters": copy.deepcopy(resolved_parameters),
            "source": source,
        },
        "target_transformation": "Original",
        "dl_category": GLOBAL_DL_CATEGORIES[model_name],
        "validation_rmse": float(validation_rmse),
        "selection_note": "Global-panel validation; reported architecture",
    }


def selection_summary(
    dataset_names: Iterable[str],
    *,
    run_ml: bool,
    ml_selections: Optional[Dict] = None,
    dl_models: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Show the predeclared ML choices and complete global DL screen."""
    rows = []
    for dataset_name in dataset_names:
        if run_ml:
            for target, record in manual_ml_records(
                dataset_name, ml_selections or {}
            ).items():
                rows.append(
                    {
                        "DGP": dataset_name,
                        "Family": "ML",
                        "Target": target,
                        "Model": record["base_model"],
                        "Selection basis": (
                            f"Manual transfer from {record['source_dataset']}"
                        ),
                    }
                )
        for model_name in dl_models or GLOBAL_DL_CATEGORIES:
            rows.append(
                {
                    "DGP": dataset_name,
                    "Family": "DL",
                    "Target": "Original",
                    "Model": model_name,
                    "DL category": GLOBAL_DL_CATEGORIES[model_name],
                    "Selection basis": "Global validation (all architectures)",
                }
            )
    return pd.DataFrame(rows)


def deterministic_partitions(
    series_ids: Iterable[str],
    partition_size: Optional[int],
    seed: int = SEED,
) -> List[List[str]]:
    """Create reproducible random workload partitions without using targets."""
    ids = np.asarray(sorted(str(value) for value in set(series_ids)), dtype=object)
    if len(ids) == 0:
        raise ValueError("No series IDs supplied")
    if partition_size is None:
        partition_size = len(ids)
    partition_size = int(partition_size)
    if partition_size < 1:
        raise ValueError("partition_size must be positive or None")
    shuffled = np.random.default_rng(seed).permutation(ids)
    return [
        shuffled[start : start + partition_size].tolist()
        for start in range(0, len(shuffled), partition_size)
    ]


def _estimator_from_record(record: Dict, seed: int, n_jobs: Optional[int]):
    config = record["config"]
    module = importlib.import_module(config["module"])
    estimator_class = getattr(module, config["class"])
    parameters = {
        key: value
        for key, value in config["parameters"].items()
        if value is not None
    }
    parameters["random_state"] = int(seed)
    if n_jobs is not None and "n_jobs" in parameters:
        parameters["n_jobs"] = int(n_jobs)
    return estimator_class(**parameters)


def _suggest_global_xgb_random_forest(trial) -> Dict:
    """Use the same XGBRandomForest search space as the local ML benchmark."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 0.5, 20.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def _global_xgb_record(
    parameters: Dict,
    lags: List[int],
    *,
    seed: int,
    n_jobs: Optional[int],
    tuning_time: float,
    validation_rmse: float,
) -> Dict:
    model_parameters = {
        "learning_rate": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": int(seed),
        "n_jobs": n_jobs,
        "verbosity": 0,
        **parameters,
    }
    return {
        "base_model": "XGBRandomForest",
        "model": "XGBRandomForest",
        "config": {
            "class": "XGBRFRegressor",
            "module": "xgboost.sklearn",
            "parameters": model_parameters,
            "forecasting": {
                "lags": [int(lag) for lag in lags],
                "strategy": "recursive",
            },
        },
        "tuning_time": float(tuning_time),
        "validation_rmse": float(validation_rmse),
        "selection_note": "Pooled global-panel validation with Optuna",
    }


def tune_global_xgb_random_forest(
    fit_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    transformation: str,
    seasonality: int,
    lags: List[int],
    partition_size: Optional[int],
    freq: str,
    n_trials: int = 25,
    seed: int = SEED,
    n_jobs: Optional[int] = None,
) -> Dict:
    """Tune one pooled XGBRandomForest per deterministic series block.

    A trial fits one estimator to the pooled lagged rows from every series in
    its block and is scored by mean series-level RMSE over the common external
    validation horizon. It never launches one Optuna study per series.
    """
    import optuna
    from xgboost import XGBRFRegressor

    try:
        from ml_evaluation import recursive_forecast
    except ImportError:
        from thesis.ml_evaluation import recursive_forecast

    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    ordered_lags = sorted({int(lag) for lag in lags})
    if not ordered_lags or ordered_lags[0] < 1:
        raise ValueError("lags must contain positive integers")

    partitions = deterministic_partitions(
        fit_df["unique_id"].unique(), partition_size, seed
    )
    partition_records = {}
    for partition_number, series_ids in enumerate(partitions, start=1):
        prepared = _prepare_partition(
            fit_df,
            series_ids,
            ordered_lags,
            transformation,
            seasonality,
            freq,
            stopping_fraction=None,
        )

        def objective(trial):
            parameters = _suggest_global_xgb_random_forest(trial)
            estimator = XGBRFRegressor(
                learning_rate=1.0,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=int(seed),
                n_jobs=n_jobs,
                verbosity=0,
                **parameters,
            )
            estimator.fit(prepared["X"], prepared["y"])
            series_scores = []
            for series_id in series_ids:
                evaluation = validation_df.loc[
                    validation_df["unique_id"].eq(series_id)
                ].sort_values("ds")
                prediction = recursive_forecast(
                    estimator,
                    prepared["histories"][series_id],
                    len(evaluation),
                    ordered_lags,
                    pd.DatetimeIndex(pd.to_datetime(evaluation["ds"])),
                )
                transformer = prepared["transformers"][series_id]
                if transformer is not None:
                    prediction = transformer.inverse_transform(prediction)
                actual = evaluation["y"].astype(float).to_numpy()
                predicted = np.asarray(prediction, dtype=float).reshape(-1)
                series_scores.append(
                    float(np.sqrt(np.mean(np.square(actual - predicted))))
                )
            return float(np.mean(series_scores))

        print(
            f"  Global ML block {partition_number}/{len(partitions)}: "
            f"{len(series_ids)} pooled series x {n_trials} Optuna trials"
        )
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=int(seed), n_startup_trials=min(5, int(n_trials))
            ),
        )
        tuning_start = time.perf_counter()
        study.optimize(objective, n_trials=int(n_trials), show_progress_bar=True)
        tuning_time = time.perf_counter() - tuning_start
        partition_records[str(partition_number)] = {
            "series_ids": list(series_ids),
            "record": _global_xgb_record(
                study.best_params,
                ordered_lags,
                seed=seed,
                n_jobs=n_jobs,
                tuning_time=tuning_time,
                validation_rmse=study.best_value,
            ),
        }

    return {
        "base_model": "XGBRandomForest",
        "model": "XGBRandomForest",
        "partition_size": partition_size,
        "partition_seed": int(seed),
        "transformation": transformation,
        "partition_configs": partition_records,
        "selection_note": "Pooled global-panel validation with Optuna",
    }


def _stationary_transformer(seasonality: int):
    # The source transformer requires seasonal_period > 1 when it is supplied.
    seasonal_period = int(seasonality) if int(seasonality) > 1 else None
    return AutoStationaryTransformer(
        seasonal_period=seasonal_period,
        deseasonalizer_params={},
    )


def _prepare_partition(
    fit_df: pd.DataFrame,
    series_ids: List[str],
    lags: List[int],
    transformation: str,
    seasonality: int,
    freq: str,
    stopping_fraction: Optional[float],
):
    try:
        from ml_evaluation import make_lagged_regression_data
    except ImportError:
        from thesis.ml_evaluation import make_lagged_regression_data

    full_X, full_y, stop_train_X, stop_train_y, stop_eval_X, stop_eval_y = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    histories = {}
    transformers = {}

    for series_id in series_ids:
        group = fit_df.loc[fit_df["unique_id"].eq(series_id)].sort_values("ds")
        index = pd.DatetimeIndex(pd.to_datetime(group["ds"]))
        original = pd.Series(
            group["y"].astype(float).to_numpy(), index=index, name="y"
        )
        transformer = None
        history = original
        if transformation == "AutoStationary":
            transformer = _stationary_transformer(seasonality)
            history = transformer.fit_transform(original, freq=freq)
        elif transformation != "Original":
            raise ValueError(f"Unsupported transformation {transformation!r}")

        X_series, y_series = make_lagged_regression_data(history, lags)
        full_X.append(X_series)
        full_y.append(y_series)
        histories[series_id] = history
        transformers[series_id] = transformer

        if stopping_fraction is not None:
            validation_size = max(1, int(np.ceil(len(X_series) * stopping_fraction)))
            split_at = len(X_series) - validation_size
            if split_at < 2:
                raise ValueError("Not enough rows for pooled temporal early stopping")
            stop_train_X.append(X_series.iloc[:split_at])
            stop_train_y.append(y_series.iloc[:split_at])
            stop_eval_X.append(X_series.iloc[split_at:])
            stop_eval_y.append(y_series.iloc[split_at:])

    prepared = {
        "X": pd.concat(full_X, ignore_index=True),
        "y": pd.concat(full_y, ignore_index=True),
        "histories": histories,
        "transformers": transformers,
    }
    if stopping_fraction is not None:
        prepared.update(
            {
                "stop_train_X": pd.concat(stop_train_X, ignore_index=True),
                "stop_train_y": pd.concat(stop_train_y, ignore_index=True),
                "stop_eval_X": pd.concat(stop_eval_X, ignore_index=True),
                "stop_eval_y": pd.concat(stop_eval_y, ignore_index=True),
            }
        )
    return prepared


def evaluate_global_ml(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    record: Dict,
    transformation: str,
    seasonality: int,
    partition_size: Optional[int] = None,
    freq: str = "D",
    seed: int = SEED,
    n_jobs: Optional[int] = None,
):
    """Fit pooled recursive tree models and score every series separately."""
    try:
        from ml_evaluation import recursive_forecast
    except ImportError:
        from thesis.ml_evaluation import recursive_forecast

    representative_record = record
    if record.get("partition_configs"):
        first_partition = record["partition_configs"].get("1")
        if first_partition is None:
            raise KeyError("Missing frozen ML configuration for partition 1")
        representative_record = first_partition["record"]
    forecasting = representative_record["config"]["forecasting"]
    lags = [int(lag) for lag in forecasting["lags"]]
    partitions = deterministic_partitions(
        fit_df["unique_id"].unique(), partition_size, seed
    )
    model_name = record["base_model"]
    forecast_column = (
        f"Global_{model_name}"
        if transformation == "Original"
        else f"Global_{model_name}_AutoStationary"
    )
    results = eval_df[["unique_id", "ds", "y"]].copy()
    results["ds"] = pd.to_datetime(results["ds"])
    results[forecast_column] = np.nan
    metric_rows = []
    timing_rows = []

    tuning = representative_record["config"].get("tuning", {})
    stopping_rounds = tuning.get("early_stopping_rounds")
    stopping_fraction = (
        float(tuning.get("early_stopping_fraction", 0.2))
        if stopping_rounds
        else None
    )

    for partition_number, series_ids in enumerate(partitions, start=1):
        partition_record = record
        if record.get("partition_configs"):
            saved_partition = record["partition_configs"].get(
                str(partition_number)
            )
            if saved_partition is None:
                raise KeyError(
                    f"Missing frozen ML configuration for partition "
                    f"{partition_number}"
                )
            if list(saved_partition.get("series_ids", [])) != list(series_ids):
                raise ValueError(
                    "Frozen ML partition membership does not match the current "
                    "dataset, partition size, and seed"
                )
            partition_record = saved_partition["record"]

        preprocessing_start = time.perf_counter()
        prepared = _prepare_partition(
            fit_df,
            series_ids,
            lags,
            transformation,
            seasonality,
            freq,
            stopping_fraction,
        )
        preprocessing_time = time.perf_counter() - preprocessing_start

        estimator = _estimator_from_record(partition_record, seed, n_jobs)
        best_iteration = None
        training_start = time.perf_counter()
        if stopping_rounds:
            stopping_model = clone(estimator)
            if stopping_model.__class__.__module__.lower().startswith("lightgbm"):
                from lightgbm import early_stopping

                stopping_model.fit(
                    prepared["stop_train_X"],
                    prepared["stop_train_y"],
                    eval_set=[(prepared["stop_eval_X"], prepared["stop_eval_y"])],
                    eval_metric="rmse",
                    callbacks=[
                        early_stopping(
                            stopping_rounds=int(stopping_rounds), verbose=False
                        )
                    ],
                )
                best_iteration = max(1, int(stopping_model.best_iteration_))
                estimator.set_params(n_estimators=best_iteration)
            else:
                raise TypeError(
                    "Early stopping is supported only for LightGBM"
                )
        estimator.fit(prepared["X"], prepared["y"])
        training_time = time.perf_counter() - training_start

        forecast_start = time.perf_counter()
        for series_id in series_ids:
            evaluation = eval_df.loc[
                eval_df["unique_id"].eq(series_id)
            ].sort_values("ds")
            future_index = pd.DatetimeIndex(pd.to_datetime(evaluation["ds"]))
            prediction = recursive_forecast(
                estimator,
                prepared["histories"][series_id],
                len(evaluation),
                lags,
                future_index,
            )
            transformer = prepared["transformers"][series_id]
            if transformer is not None:
                prediction = transformer.inverse_transform(prediction)
            target_index = results.loc[
                results["unique_id"].eq(series_id)
            ].sort_values("ds").index
            results.loc[target_index, forecast_column] = prediction.to_numpy()
        forecast_time = time.perf_counter() - forecast_start

        timing_rows.append(
            {
                "Model Family": "ML",
                "Model": forecast_column,
                "Target Transformation": transformation,
                "Partition": partition_number,
                "Series in Partition": len(series_ids),
                "Preprocessing Time": preprocessing_time,
                "Tuning Time": float(partition_record.get("tuning_time", 0.0)),
                "Training Time": training_time,
                "Forecast Time": forecast_time,
                "Best Iteration": best_iteration,
                "Time Elapsed": preprocessing_time + training_time + forecast_time,
            }
        )

    if results[forecast_column].isna().any():
        raise ValueError(f"Missing forecasts in {forecast_column}")

    partition_lookup = {
        series_id: partition_number
        for partition_number, ids in enumerate(partitions, start=1)
        for series_id in ids
    }
    for series_id, group in results.groupby("unique_id", sort=False):
        ordered = group.sort_values("ds")
        actual = ordered["y"].astype(float).to_numpy()
        predicted = ordered[forecast_column].astype(float).to_numpy()
        metric_rows.append(
            {
                "unique_id": series_id,
                "Model": forecast_column,
                "Base Model": model_name,
                "Model Family": "ML",
                "Target Transformation": transformation,
                "Partition": partition_lookup[series_id],
                "mae": float(np.mean(np.abs(actual - predicted))),
                "mse": float(np.mean(np.square(actual - predicted))),
                "rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
                "forecast_bias": _forecast_bias(actual, predicted),
            }
        )

    return (
        results.sort_values(["unique_id", "ds"]).reset_index(drop=True),
        pd.DataFrame(metric_rows),
        pd.DataFrame(timing_rows),
    )


def evaluate_global_dl(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    record: Dict,
    freq: str = "D",
    progress_label: str = "",
    *,
    tune: bool = False,
    internal_val_size: int = 0,
):
    """Tune or refit one global NeuralForecast model shared by all series."""
    try:
        from dl_evaluation import evaluate_dl_models
    except ImportError:
        from thesis.dl_evaluation import evaluate_dl_models

    model_name = record["base_model"]
    saved_config = record["config"]
    if tune:
        config = copy.deepcopy(saved_config)
        config["_source"] = saved_config.get(
            "_source", "Global-panel Auto validation"
        )
    else:
        config = dict(saved_config.get("parameters", saved_config))
        config["_source"] = saved_config.get(
            "source", "Frozen global-panel validation configuration"
        )
    results, metrics, _, resolved = evaluate_dl_models(
        fit_df=fit_df,
        eval_df=eval_df,
        configs={model_name: config},
        freq=freq,
        internal_val_size=int(internal_val_size),
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        retain_fitted_models=False,
        tune=bool(tune),
        progress_label=progress_label,
    )
    metrics["Model Family"] = "DL"
    metrics["DL Category"] = record.get(
        "dl_category", GLOBAL_DL_CATEGORIES[model_name]
    )
    metrics["Selection Note"] = record.get(
        "selection_note", "Global-panel validation architecture"
    )
    timing = (
        metrics[
            [
                "Model",
                "Target Transformation",
                "Preprocessing Time",
                "Tuning Time",
                "Training Time",
                "Forecast Time",
                "Time Elapsed",
            ]
        ]
        .head(1)
        .copy()
    )
    timing.insert(0, "Model Family", "DL")
    timing["DL Category"] = record.get(
        "dl_category", GLOBAL_DL_CATEGORIES[model_name]
    )
    timing["Partition"] = 1
    timing["Series in Partition"] = fit_df["unique_id"].nunique()
    return results, metrics, timing, resolved


def aggregate_global_metrics(
    metrics: pd.DataFrame,
    dataset_name: str,
    evaluation_split: str,
) -> pd.DataFrame:
    """Average equal-horizon metrics across all global panel members."""
    group_columns = [
        "Model Family",
        "Model",
        "Base Model",
        "Target Transformation",
    ]
    for optional_column in ("DL Category",):
        if optional_column in metrics.columns:
            group_columns.append(optional_column)
    table = (
        metrics.groupby(group_columns, dropna=False)[
            ["mae", "mse", "rmse", "forecast_bias"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "mae": "Mean MAE",
                "mse": "Mean MSE",
                "rmse": "Mean RMSE",
                "forecast_bias": "Mean Forecast Bias",
            }
        )
    )
    table.insert(0, "DGP", dataset_name)
    table.insert(1, "Split", evaluation_split)
    table.insert(2, "Number of Series", metrics["unique_id"].nunique())
    return table.sort_values("Mean RMSE").reset_index(drop=True)
