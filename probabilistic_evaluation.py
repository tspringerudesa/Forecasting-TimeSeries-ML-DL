"""Utilities for the S&P 500 adaptive-conformal forecasting exercise.

The module deliberately keeps model selection, conformal calibration and final
testing as three chronological stages.  Point forecasts are always evaluated on
the original log-return scale, including forecasts produced after fitting an
AutoStationaryTransformer.
"""

from __future__ import annotations

import copy
import contextlib
import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.transforms.target_transformations import AutoStationaryTransformer
from thesis.runtime import CPU_THREADS

DEFAULT_CLASSICAL_MODELS = (
    "ZeroReturn",
    "HistoricAverage",
    "Naive",
    "SeasonalNaive5",
)
DEFAULT_ML_MODELS = ("XGBRandomForest", "LightGBM")
DEFAULT_PARAMETRIC_CLASSICAL_MODELS = (
    "AutoETS",
    "AutoARIMA",
    "AutoTheta",
)

# Lightning may attach handlers to the active stdout/stderr stream.  A
# process-lifetime NUL stream keeps those handlers valid after redirection
# ends, unlike a short-lived temporary file that caused closed-stream errors.
_QUIET_STREAM = open(os.devnull, "w", encoding="utf-8")


def _remove_closed_logging_handlers() -> None:
    """Remove handlers left pointing at closed notebook capture streams."""
    loggers = [logging.getLogger()]
    loggers.extend(
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )
    for logger in loggers:
        for handler in list(logger.handlers):
            stream = getattr(handler, "stream", None)
            if stream is not None and getattr(stream, "closed", False):
                logger.removeHandler(handler)


def load_sp500_returns(csv_path: str | Path) -> pd.DataFrame:
    """Load the exported S&P 500 data and construct close-to-close log returns."""
    frame = pd.read_csv(csv_path)
    date_candidates = ["date", "Date", "indice_tiempo", "ds"]
    close_candidates = ["sp500", "SP500", "close", "Close", "value", "y"]
    date_col = next((column for column in date_candidates if column in frame), None)
    close_col = next((column for column in close_candidates if column in frame), None)
    if date_col is None or close_col is None:
        raise ValueError(
            "The S&P file must contain a date column and a close/value column. "
            f"Found columns: {list(frame.columns)}"
        )

    requested_columns = [date_col, close_col]
    return_col = next(
        (
            column
            for column in ("log_return", "return", "returns")
            if column in frame
        ),
        None,
    )
    if return_col:
        requested_columns.append(return_col)
    output = frame[requested_columns].rename(
        columns={date_col: "date", close_col: "close"}
    )
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["close"] = pd.to_numeric(output["close"], errors="coerce")
    output = (
        output.dropna()
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if return_col:
        output["y"] = pd.to_numeric(output[return_col], errors="coerce")
    else:
        output["y"] = np.log(output["close"]).diff()
    output = output.dropna(subset=["y"]).reset_index(drop=True)
    output["unique_id"] = "SP500"
    # NeuralForecast needs a regular frequency.  Consecutive trading-session
    # ordinals avoid pretending that exchange holidays are missing observations.
    output["ds"] = np.arange(len(output), dtype=np.int64)
    return output[["unique_id", "ds", "date", "close", "y"]]


def split_sp500_stages(
    frame: pd.DataFrame,
    selection_year: int = 2023,
    calibration_year: int = 2024,
    test_year: int = 2025,
) -> dict[str, pd.DataFrame]:
    """Return chronological train, selection, calibration and test partitions."""
    partitions = {
        "train": frame[frame["date"].dt.year < selection_year],
        "selection": frame[frame["date"].dt.year == selection_year],
        "calibration": frame[frame["date"].dt.year == calibration_year],
        "test": frame[frame["date"].dt.year == test_year],
    }
    empty = [name for name, part in partitions.items() if part.empty]
    if empty:
        raise ValueError(f"Empty S&P partition(s): {empty}")
    return {name: part.reset_index(drop=True) for name, part in partitions.items()}


def point_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return {
        "mae": float(mean_absolute_error(actual, forecast)),
        "mse": float(mean_squared_error(actual, forecast)),
        "rmse": float(np.sqrt(mean_squared_error(actual, forecast))),
    }


def _regular_index(steps: pd.Series | np.ndarray) -> pd.DatetimeIndex:
    origin = pd.Timestamp("2000-01-01")
    return origin + pd.to_timedelta(np.asarray(steps, dtype=np.int64), unit="D")


def prepare_target_representation(
    fit_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    transformation: str,
    seasonal_period: int = 5,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Fit a stage-local transformation without using future observations."""
    fit_series = pd.Series(
        fit_df["y"].to_numpy(dtype=float),
        index=_regular_index(fit_df["ds"]),
        name="y",
    )
    evaluation_series = pd.Series(
        evaluation_df["y"].to_numpy(dtype=float),
        index=_regular_index(evaluation_df["ds"]),
        name="y",
    )
    if transformation == "Original":
        return fit_series.to_numpy(), evaluation_series.to_numpy(), None
    if transformation != "AutoStationary":
        raise ValueError(f"Unknown target transformation: {transformation}")

    transformer = AutoStationaryTransformer(
        **(
            {"seasonal_period": int(seasonal_period)}
            if int(seasonal_period) > 1
            else {}
        )
    )
    transformer.fit(fit_series, freq="D")
    pipeline_names = [
        type(step).__name__ for step in getattr(transformer, "_pipeline", [])
    ]
    # AutoStationary makes a signed intermediate series Box-Cox compatible by
    # estimating AddM from the fitting minimum.  A later observation can be
    # lower than that minimum, so the frozen shift cannot guarantee positive
    # out-of-sample values.  Looking at the holdout to enlarge the shift would
    # leak information.  For this case retain the fitted trend/seasonal steps
    # but omit the AddM+BoxCox pair; every retained step remains causal and
    # exactly invertible.  Direct Box-Cox on a naturally positive target is
    # unaffected.
    if "AddMTransformer" in pipeline_names and "BoxCoxTransformer" in pipeline_names:
        transformer._pipeline = [
            step
            for step in transformer._pipeline
            if type(step).__name__
            not in {"AddMTransformer", "BoxCoxTransformer"}
        ]
        transformer._oos_boxcox_skipped = True
    else:
        transformer._oos_boxcox_skipped = False
    transformed_fit = transformer.transform(fit_series)
    transformed_evaluation = (
        transformer.transform(evaluation_series)
        if len(evaluation_series)
        else evaluation_series
    )
    return (
        np.asarray(transformed_fit, dtype=float),
        np.asarray(transformed_evaluation, dtype=float),
        transformer,
    )


def _inverse_forecast(
    forecast: float,
    transformer: Any,
    step: int,
) -> float:
    if transformer is None:
        return float(forecast)
    transformed = pd.Series(
        [float(forecast)], index=_regular_index(np.array([step])), name="y"
    )
    inverted = transformer.inverse_transform(transformed)
    return float(np.asarray(inverted, dtype=float)[0])


def _inverse_forecasts(
    forecasts: np.ndarray,
    transformer: Any,
    steps: np.ndarray,
) -> np.ndarray:
    if transformer is None:
        return np.asarray(forecasts, dtype=float)
    transformed = pd.Series(
        np.asarray(forecasts, dtype=float),
        index=_regular_index(steps),
        name="y",
    )
    return np.asarray(transformer.inverse_transform(transformed), dtype=float)


def _classical_point_forecast(
    model_name: str,
    history: list[float],
    seasonal_period: int = 5,
) -> float:
    if model_name == "ZeroReturn":
        return 0.0
    if model_name == "HistoricAverage":
        return float(np.mean(history))
    if model_name == "Naive":
        return float(history[-1])
    if model_name == "SeasonalNaive5":
        return float(history[-5] if len(history) >= 5 else history[-1])
    if model_name == "SeasonalNaive":
        lag = max(1, int(seasonal_period))
        return float(history[-lag] if len(history) >= lag else history[-1])
    if model_name == "AR1OLS":
        if len(history) < 3:
            return float(history[-1])
        lagged = np.asarray(history[:-1], dtype=float)
        current = np.asarray(history[1:], dtype=float)
        design = np.column_stack([np.ones(len(lagged)), lagged])
        intercept, coefficient = np.linalg.lstsq(
            design, current, rcond=None
        )[0]
        return float(intercept + coefficient * history[-1])
    raise ValueError(f"Unknown classical model: {model_name}")


def forecast_classical_stage(
    fit_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    model_name: str,
    transformation: str,
    seasonal_period: int = 5,
) -> pd.DataFrame:
    fit_y, evaluation_y, transformer = prepare_target_representation(
        fit_df,
        evaluation_df,
        transformation,
        seasonal_period=seasonal_period,
    )
    history = fit_y.tolist()
    predictions: list[float] = []
    for row, observed_transformed in zip(
        evaluation_df.itertuples(index=False), evaluation_y
    ):
        transformed_forecast = _classical_point_forecast(
            model_name,
            history,
            seasonal_period=seasonal_period,
        )
        predictions.append(
            _inverse_forecast(transformed_forecast, transformer, int(row.ds))
        )
        # One-step rolling forecasting may use the newly observed return at the
        # next origin, but never uses it before the current forecast is made.
        history.append(float(observed_transformed))
    return _prediction_frame(evaluation_df, predictions, model_name, transformation)


def _make_parametric_classical_model(
    model_name: str,
    season_length: int = 5,
):
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

    if model_name == "AutoETS":
        return AutoETS(model="AAA", season_length=season_length)
    if model_name == "AutoARIMA":
        return AutoARIMA(
            max_p=8,
            max_q=8,
            max_P=3,
            max_Q=3,
            max_order=15,
            max_d=2,
            max_D=1,
            seasonal=season_length > 1,
            stepwise=True,
            nmodels=200,
            season_length=season_length,
        )
    if model_name == "AutoTheta":
        return AutoTheta(season_length=season_length)
    raise ValueError(f"Unknown parametric classical model: {model_name}")


def forecast_parametric_classical_stage(
    fit_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    model_name: str,
    transformation: str,
    season_length: int = 5,
) -> tuple[pd.DataFrame, float]:
    """Issue rolling one-step StatsForecast predictions without daily refits.

    ``cross_validation(..., h=1, refit=False)`` fits the model at the first
    origin and then applies its ``forward`` method as observations arrive.
    """
    from statsforecast import StatsForecast

    fit_y, evaluation_y, transformer = prepare_target_representation(
        fit_df, evaluation_df, transformation, seasonal_period=season_length
    )
    combined = pd.DataFrame(
        {
            "unique_id": pd.concat(
                [fit_df["unique_id"], evaluation_df["unique_id"]],
                ignore_index=True,
            ),
            "ds": np.arange(len(fit_df) + len(evaluation_df), dtype=np.int64),
            "y": np.concatenate([fit_y, evaluation_y]),
        }
    )
    model = _make_parametric_classical_model(model_name, season_length)
    forecaster = StatsForecast(models=[model], freq=1, n_jobs=CPU_THREADS)
    started = time.perf_counter()
    rolling = forecaster.cross_validation(
        df=combined,
        h=1,
        n_windows=len(evaluation_df),
        step_size=1,
        refit=False,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )
    elapsed = time.perf_counter() - started
    if not isinstance(rolling, pd.DataFrame):
        rolling = rolling.to_pandas()
    prediction_columns = [
        column
        for column in rolling.columns
        if column not in {"unique_id", "ds", "cutoff", "y"}
    ]
    if len(prediction_columns) != 1:
        raise ValueError(
            f"Could not identify {model_name}'s forecast column: "
            f"{prediction_columns}"
        )
    ordered = rolling.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    if len(ordered) != len(evaluation_df):
        raise ValueError(
            f"{model_name} returned {len(ordered)} rolling forecasts; "
            f"expected {len(evaluation_df)}."
        )
    predictions = _inverse_forecasts(
        ordered[prediction_columns[0]].to_numpy(dtype=float),
        transformer,
        evaluation_df["ds"].to_numpy(dtype=np.int64),
    )
    return (
        _prediction_frame(
            evaluation_df, predictions, model_name, transformation
        ),
        elapsed,
    )


def _prediction_frame(
    evaluation_df: pd.DataFrame,
    predictions: list[float] | np.ndarray,
    model_name: str,
    transformation: str,
) -> pd.DataFrame:
    label = (
        model_name
        if transformation == "Original"
        else f"{model_name}_AutoStationary"
    )
    return pd.DataFrame(
        {
            "unique_id": evaluation_df["unique_id"].to_numpy(),
            "ds": evaluation_df["ds"].to_numpy(),
            "date": evaluation_df["date"].to_numpy(),
            "y": evaluation_df["y"].to_numpy(dtype=float),
            "Model": label,
            "Base Model": model_name,
            "Target Transformation": transformation,
            "y_hat": np.asarray(predictions, dtype=float),
        }
    )


def _make_ml_estimator(
    model_name: str,
    params: dict[str, Any],
    seed: int,
):
    if model_name == "XGBRandomForest":
        from xgboost import XGBRFRegressor

        defaults = {
            "objective": "reg:squarederror",
            "random_state": seed,
            "n_jobs": CPU_THREADS,
            "verbosity": 0,
        }
        return XGBRFRegressor(**defaults, **params)
    if model_name == "LightGBM":
        from lightgbm import LGBMRegressor

        defaults = {
            "objective": "regression",
            "random_state": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
            "force_col_wise": True,
            "subsample_freq": 1,
            "n_jobs": CPU_THREADS,
            "verbosity": -1,
        }
        return LGBMRegressor(**defaults, **params)
    raise ValueError(f"Unknown ML model: {model_name}")


def _suggest_ml_params(trial: Any, model_name: str) -> dict[str, Any]:
    if model_name == "XGBRandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 30.0, log=True),
            "learning_rate": 1.0,
        }
    if model_name == "LightGBM":
        max_depth = trial.suggest_int("max_depth", 2, 10)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.15, log=True
            ),
            "max_depth": max_depth,
            "num_leaves": trial.suggest_int(
                "num_leaves", 4, min(2**max_depth, 128)
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 100, step=10
            ),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.55, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 30.0, log=True),
        }
    raise ValueError(f"Unknown ML model: {model_name}")


def tune_ml_candidate(
    fit_df: pd.DataFrame,
    model_name: str,
    transformation: str,
    lags: list[int],
    n_trials: int,
    seed: int,
    n_splits: int = 3,
    seasonal_period: int = 5,
) -> tuple[dict[str, Any], float]:
    """Tune one tree model with ordered expanding-window validation folds."""
    import optuna
    from thesis.ml_evaluation import make_lagged_regression_data

    started = time.perf_counter()
    splitter = TimeSeriesSplit(n_splits=n_splits)
    ordered_lags = sorted(set(lags))
    feature_columns = [f"target_lag_{lag}" for lag in ordered_lags]
    # Transformation parameters are fold-specific.  Precomputing the fold
    # matrices is safe because the same chronological folds are used for every
    # Optuna trial and no fold transformer sees its validation observations.
    fold_datasets = []
    for train_index, validation_index in splitter.split(fit_df):
        fold_fit = fit_df.iloc[train_index]
        fold_validation = fit_df.iloc[validation_index]
        transformed_fit, transformed_validation, transformer = (
            prepare_target_representation(
                fold_fit,
                fold_validation,
                transformation,
                seasonal_period=seasonal_period,
            )
        )
        X_train, y_train = make_lagged_regression_data(
            pd.Series(transformed_fit, name="y"), ordered_lags
        )
        history = transformed_fit.tolist()
        validation_rows = []
        for observed in transformed_validation:
            validation_rows.append([history[-lag] for lag in ordered_lags])
            history.append(float(observed))
        X_validation = pd.DataFrame(
            validation_rows, columns=feature_columns
        )
        fold_datasets.append(
            (
                X_train,
                y_train,
                X_validation,
                fold_validation["y"].to_numpy(dtype=float),
                fold_validation["ds"].to_numpy(dtype=np.int64),
                transformer,
            )
        )

    def objective(trial: Any) -> float:
        params = _suggest_ml_params(trial, model_name)
        fold_rmse: list[float] = []
        for (
            X_train,
            y_train,
            X_validation,
            actual,
            validation_steps,
            transformer,
        ) in fold_datasets:
            estimator = _make_ml_estimator(model_name, params, seed)
            estimator.fit(X_train, y_train)
            transformed_forecast = estimator.predict(X_validation)
            forecast = _inverse_forecasts(
                transformed_forecast, transformer, validation_steps
            )
            fold_rmse.append(
                float(np.sqrt(mean_squared_error(actual, forecast)))
            )
        return float(np.mean(fold_rmse))

    sampler = optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=min(3, n_trials)
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.perf_counter() - started
    best_params = study.best_params.copy()
    if model_name == "XGBRandomForest":
        best_params["learning_rate"] = 1.0
    return best_params, elapsed


def forecast_ml_stage(
    fit_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    model_name: str,
    transformation: str,
    params: dict[str, Any],
    lags: list[int],
    seed: int,
    seasonal_period: int = 5,
) -> tuple[pd.DataFrame, float]:
    """Fit a tree once, then issue rolling one-step forecasts using observed lags."""
    from thesis.ml_evaluation import make_lagged_regression_data

    fit_y, evaluation_y, transformer = prepare_target_representation(
        fit_df,
        evaluation_df,
        transformation,
        seasonal_period=seasonal_period,
    )
    X, y = make_lagged_regression_data(pd.Series(fit_y, name="y"), lags)
    estimator = _make_ml_estimator(model_name, params, seed)
    started = time.perf_counter()
    estimator.fit(X, y)

    history = fit_y.tolist()
    predictions: list[float] = []
    ordered_lags = list(lags)
    for row, observed_transformed in zip(
        evaluation_df.itertuples(index=False), evaluation_y
    ):
        feature_values = np.array(
            [[history[-lag] for lag in ordered_lags]], dtype=float
        )
        feature_frame = pd.DataFrame(
            feature_values, columns=[f"target_lag_{lag}" for lag in ordered_lags]
        )
        transformed_forecast = float(estimator.predict(feature_frame)[0])
        predictions.append(
            _inverse_forecast(transformed_forecast, transformer, int(row.ds))
        )
        history.append(float(observed_transformed))
    del estimator
    gc.collect()
    elapsed = time.perf_counter() - started
    return (
        _prediction_frame(
            evaluation_df, predictions, model_name, transformation
        ),
        elapsed,
    )


def score_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for frame in frames:
        metric_values = point_metrics(frame["y"], frame["y_hat"])
        rows.append(
            {
                "Model": frame["Model"].iloc[0],
                "Base Model": frame["Base Model"].iloc[0],
                "Target Transformation": frame[
                    "Target Transformation"
                ].iloc[0],
                **metric_values,
            }
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def adapt_one_step_dl_configs(
    config_dir: str | Path,
    model_names: list[str],
    input_size_candidates: list[int],
    num_samples: int,
    max_steps: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Load thesis Auto-model configs for a one-step rolling experiment."""
    from thesis.dl_evaluation import load_dl_configs

    configs = load_dl_configs(config_dir, model_names=model_names)
    adapted: dict[str, dict[str, Any]] = {}
    for model_name, original in configs.items():
        config = copy.deepcopy(original)
        auto = config.setdefault("_auto", {})
        auto["num_samples"] = int(num_samples)
        auto["n_startup_trials"] = min(2, int(num_samples))
        auto["seed"] = int(seed)
        config["_input_size_candidates"] = [
            int(value) for value in input_size_candidates
        ]
        config["max_steps"] = int(max_steps)
        config.setdefault("val_check_steps", 10)
        config.setdefault("early_stop_patience_steps", 20)
        config["random_seed"] = int(seed)
        # These are forwarded by NeuralForecast models to Lightning's Trainer.
        # Without them every Optuna trial and every rolling one-step predict
        # call emits a separate progress bar/model summary in Jupyter.
        config["enable_progress_bar"] = False
        config["enable_model_summary"] = False
        config["enable_checkpointing"] = False
        config["logger"] = False
        if model_name == "AutoNBEATS":
            # NeuralForecast's default trend/seasonality stacks require a
            # horizon greater than one.  Three generic stacks retain the
            # N-BEATS architecture for one-step conformal exercises.
            config["stack_types"] = ["identity", "identity", "identity"]
        adapted[model_name] = config
    return adapted


def adapt_daily_dl_configs(
    config_dir: str | Path,
    model_names: list[str],
    input_size_candidates: list[int],
    num_samples: int,
    max_steps: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Backward-compatible name used by the S&P 500 daily notebook."""
    configs = adapt_one_step_dl_configs(
        config_dir,
        model_names,
        input_size_candidates,
        num_samples,
        max_steps,
        seed,
    )
    for config in configs.values():
        config["val_check_steps"] = min(50, max(10, max_steps // 10))
        config["early_stop_patience_steps"] = 5
    return configs


def _nf_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["unique_id", "ds", "y"]].copy()


def sequential_neural_forecast(
    neural_forecast: Any,
    fit_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Forecast each next session while revealing its return only afterwards."""
    history = _nf_frame(fit_df)
    predictions: list[float] = []
    for row in evaluation_df.itertuples(index=False):
        future = pd.DataFrame(
            {"unique_id": [row.unique_id], "ds": [int(row.ds)]}
        )
        forecast = neural_forecast.predict(df=history, futr_df=future)
        candidate_columns = [
            column
            for column in forecast.columns
            if column not in {"unique_id", "ds"}
        ]
        exact = next(
            (column for column in candidate_columns if column == model_name), None
        )
        prediction_column = exact or candidate_columns[0]
        predictions.append(float(forecast[prediction_column].iloc[0]))
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    {
                        "unique_id": [row.unique_id],
                        "ds": [int(row.ds)],
                        "y": [float(row.y)],
                    }
                ),
            ],
            ignore_index=True,
        )
    return _prediction_frame(
        evaluation_df, predictions, model_name, "Original"
    )


def validate_dl_candidates(
    fit_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
    internal_val_size: int,
) -> tuple[list[pd.DataFrame], pd.DataFrame, dict[str, dict[str, Any]]]:
    """Tune and score each Auto model, releasing it before the next candidate."""
    from thesis.dl_evaluation import evaluate_dl_models
    import optuna

    _remove_closed_logging_handlers()
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    resolved_configs: dict[str, dict[str, Any]] = {}
    previous_optuna_verbosity = optuna.logging.get_verbosity()
    lightning_loggers = [
        logging.getLogger("lightning"),
        logging.getLogger("lightning.pytorch"),
        logging.getLogger("lightning_fabric"),
        logging.getLogger("pytorch_lightning"),
    ]
    previous_lightning_levels = [logger.level for logger in lightning_loggers]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for logger in lightning_loggers:
        logger.setLevel(logging.ERROR)
    for model_name, config in configs.items():
        started = time.perf_counter()
        print(f"DL selection candidate {model_name} started.")
        try:
            with contextlib.redirect_stdout(
                _QUIET_STREAM
            ), contextlib.redirect_stderr(_QUIET_STREAM):
                _, _, fitted, resolved = evaluate_dl_models(
                    _nf_frame(fit_df),
                    _nf_frame(selection_df.iloc[:1]),
                    {model_name: config},
                    freq=1,
                    internal_val_size=internal_val_size,
                    id_col="unique_id",
                    time_col="ds",
                    target_col="y",
                    tune=True,
                    retain_fitted_models=True,
                )
                neural_forecast = fitted[model_name]
                prediction_frame = sequential_neural_forecast(
                    neural_forecast, fit_df, selection_df, model_name
                )
            elapsed = time.perf_counter() - started
            metrics = point_metrics(
                prediction_frame["y"], prediction_frame["y_hat"]
            )
            metric_rows.append(
                {
                    "Model": model_name,
                    "Base Model": model_name,
                    "Target Transformation": "Original",
                    **metrics,
                    "selection_tuning_and_fit_seconds": elapsed,
                    "Status": "Completed",
                    "Error": "",
                }
            )
            prediction_frames.append(prediction_frame)
            resolved_configs[model_name] = resolved[model_name]
            del neural_forecast, fitted
            print(
                f"DL selection candidate {model_name} completed in "
                f"{elapsed:.1f}s | RMSE={metrics['rmse']:.6f}."
            )
        except Exception as error:
            elapsed = time.perf_counter() - started
            metric_rows.append(
                {
                    "Model": model_name,
                    "Base Model": model_name,
                    "Target Transformation": "Original",
                    "mae": np.nan,
                    "mse": np.nan,
                    "rmse": np.nan,
                    "selection_tuning_and_fit_seconds": elapsed,
                    "Status": "Failed",
                    "Error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"{model_name} failed and was skipped: {type(error).__name__}: {error}")
        finally:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    optuna.logging.set_verbosity(previous_optuna_verbosity)
    for logger, previous_level in zip(
        lightning_loggers, previous_lightning_levels
    ):
        logger.setLevel(previous_level)
    return (
        prediction_frames,
        pd.DataFrame(metric_rows).sort_values("rmse").reset_index(drop=True),
        resolved_configs,
    )


def forecast_fixed_dl_stage(
    fit_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    model_name: str,
    resolved_config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, float]:
    """Refit the validation-selected base architecture without re-running Optuna."""
    from neuralforecast import NeuralForecast
    from thesis.dl_evaluation import build_fixed_dl_model

    _remove_closed_logging_handlers()
    lightning_loggers = [
        logging.getLogger("lightning"),
        logging.getLogger("lightning.pytorch"),
        logging.getLogger("lightning_fabric"),
        logging.getLogger("pytorch_lightning"),
    ]
    previous_lightning_levels = [logger.level for logger in lightning_loggers]
    for logger in lightning_loggers:
        logger.setLevel(logging.ERROR)
    config = copy.deepcopy(resolved_config)
    config["random_seed"] = seed
    model = build_fixed_dl_model(
        model_name=model_name,
        horizon=1,
        config=config,
        n_series=1,
    )
    neural_forecast = NeuralForecast(models=[model], freq=1)
    print(
        f"{model_name}: fixed fit plus {len(evaluation_df)} rolling forecasts started."
    )
    started = time.perf_counter()
    with contextlib.redirect_stdout(
        _QUIET_STREAM
    ), contextlib.redirect_stderr(_QUIET_STREAM):
        neural_forecast.fit(
            df=_nf_frame(fit_df),
            val_size=0,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            verbose=False,
        )
        fit_seconds = time.perf_counter() - started
        prediction_frame = sequential_neural_forecast(
            neural_forecast, fit_df, evaluation_df, model_name
        )
    print(
        f"{model_name}: fixed fit plus rolling forecasts completed in "
        f"{time.perf_counter() - started:.1f}s."
    )
    del neural_forecast, model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    for logger, previous_level in zip(
        lightning_loggers, previous_lightning_levels
    ):
        logger.setLevel(previous_level)
    return prediction_frame, fit_seconds


def save_selection_config(
    path: str | Path,
    configuration: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(configuration, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _wide_point_forecasts(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base = frames[0][["unique_id", "ds", "date", "y"]].copy()
    for frame in frames:
        label = frame["Model"].iloc[0]
        base[label] = frame["y_hat"].to_numpy(dtype=float)
    return base


def conformalize_selected_models(
    calibration_frames: list[pd.DataFrame],
    test_frames: list[pd.DataFrame],
    level: int = 90,
    gamma: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply fixed, online fixed-alpha, and adaptive residual intervals."""
    from src.conformal.conformal_predictions import (
        ConformalPrediction,
        OnlineAdaptiveConformalInference,
    )

    calibration = _wide_point_forecasts(calibration_frames)
    test = _wide_point_forecasts(test_frames)
    interval_outputs: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    model_labels = [frame["Model"].iloc[0] for frame in calibration_frames]
    for model_label in model_labels:
        calibration_input = calibration[
            ["unique_id", "ds", "date", "y", model_label]
        ].copy()
        test_input = test[["unique_id", "ds", "date", "y", model_label]].copy()

        conformal = ConformalPrediction(
            model=model_label, level=level, alias=model_label
        )
        conformal.fit(calibration_input)
        split_output = conformal.predict(test_input.copy())
        split_lo = f"{model_label}-CP-lo-{level}"
        split_hi = f"{model_label}-CP-hi-{level}"
        split_frame = _standardize_interval_frame(
            split_output,
            model_label,
            "Split conformal",
            split_lo,
            split_hi,
        )
        interval_outputs.append(split_frame)
        summary_rows.append(
            _interval_metric_row(split_frame, level, gamma=np.nan)
        )

        online_base = ConformalPrediction(
            model=model_label, level=level, alias=model_label
        )
        online_fixed = OnlineAdaptiveConformalInference(
            online_base, gamma=0.0
        )
        online_fixed.fit(calibration_input)
        online_output = online_fixed.offline_predict(test_input.copy())
        online_lo = f"{model_label}-CP_ACI-lo-{level}"
        online_hi = f"{model_label}-CP_ACI-hi-{level}"
        online_frame = _standardize_interval_frame(
            online_output,
            model_label,
            "Online fixed-alpha",
            online_lo,
            online_hi,
        )
        interval_outputs.append(online_frame)
        summary_rows.append(
            _interval_metric_row(online_frame, level, gamma=0.0)
        )

        aci_base = ConformalPrediction(
            model=model_label, level=level, alias=model_label
        )
        adaptive = OnlineAdaptiveConformalInference(aci_base, gamma=gamma)
        adaptive.fit(calibration_input)
        aci_output = adaptive.offline_predict(test_input.copy())
        aci_lo = f"{model_label}-CP_ACI-lo-{level}"
        aci_hi = f"{model_label}-CP_ACI-hi-{level}"
        aci_frame = _standardize_interval_frame(
            aci_output,
            model_label,
            "Adaptive conformal inference",
            aci_lo,
            aci_hi,
        )
        interval_outputs.append(aci_frame)
        summary_rows.append(_interval_metric_row(aci_frame, level, gamma=gamma))

    return (
        pd.concat(interval_outputs, ignore_index=True),
        pd.DataFrame(summary_rows).sort_values(
            ["Model", "Method"]
        ).reset_index(drop=True),
    )


def _standardize_interval_frame(
    frame: pd.DataFrame,
    model_label: str,
    method: str,
    lower_column: str,
    upper_column: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": frame["unique_id"],
            "ds": frame["ds"],
            "date": frame["date"],
            "y": frame["y"],
            "Model": model_label,
            "Method": method,
            "y_hat": frame[model_label],
            "lower": frame[lower_column],
            "upper": frame[upper_column],
        }
    )


def _interval_metric_row(
    frame: pd.DataFrame,
    level: int,
    gamma: float,
) -> dict[str, Any]:
    alpha = 1.0 - level / 100.0
    actual = frame["y"].to_numpy(dtype=float)
    lower = frame["lower"].to_numpy(dtype=float)
    upper = frame["upper"].to_numpy(dtype=float)
    width = upper - lower
    interval_score = (
        width
        + (2.0 / alpha) * np.maximum(lower - actual, 0.0)
        + (2.0 / alpha) * np.maximum(actual - upper, 0.0)
    )
    return {
        "Model": frame["Model"].iloc[0],
        "Method": frame["Method"].iloc[0],
        "Nominal Coverage": level / 100.0,
        "Empirical Coverage": float(
            np.mean((actual >= lower) & (actual <= upper))
        ),
        "Coverage Error": float(
            abs(np.mean((actual >= lower) & (actual <= upper)) - level / 100.0)
        ),
        "Mean Interval Width": float(np.mean(width)),
        "Mean Interval Score": float(np.mean(interval_score)),
        "MAE": float(mean_absolute_error(actual, frame["y_hat"])),
        "RMSE": float(
            np.sqrt(mean_squared_error(actual, frame["y_hat"]))
        ),
        "ACI gamma": gamma,
    }


def plot_interval_forecast(
    interval_frame: pd.DataFrame,
    model: str,
    method: str = "Adaptive conformal inference",
    tail: int = 100,
    y_label: str = "Daily log return",
    observed_label: str = "Observed return",
    zero_line: bool = True,
    model_color: str = "tab:blue",
):
    """Plot the most recent interval origins without obscuring observations."""
    import matplotlib.pyplot as plt

    subset = interval_frame[
        (interval_frame["Model"] == model)
        & (interval_frame["Method"] == method)
    ].tail(tail)
    if subset.empty:
        raise ValueError(f"No interval rows found for {model!r} and {method!r}")
    raw_x = subset["date"]
    first_x = raw_x.iloc[0]
    if isinstance(first_x, (pd.Timestamp, np.datetime64)):
        plot_x = pd.to_datetime(raw_x, errors="raise").to_numpy()
    else:
        plot_x = pd.to_numeric(raw_x, errors="raise").to_numpy(dtype=float)
    lower = pd.to_numeric(subset["lower"], errors="raise").to_numpy(dtype=float)
    upper = pd.to_numeric(subset["upper"], errors="raise").to_numpy(dtype=float)
    observed = pd.to_numeric(subset["y"], errors="raise").to_numpy(dtype=float)
    point_forecast = pd.to_numeric(
        subset["y_hat"], errors="raise"
    ).to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(14, 5))
    axis.fill_between(
        plot_x,
        lower,
        upper,
        color=model_color,
        alpha=0.18,
        label="Prediction interval",
    )
    axis.plot(
        plot_x,
        observed,
        color="black",
        linewidth=1.0,
        label=observed_label,
    )
    axis.plot(
        plot_x,
        point_forecast,
        color=model_color,
        linewidth=1.1,
        label="Point forecast",
    )
    if zero_line:
        axis.axhline(0.0, color="grey", linewidth=0.8, alpha=0.6)
    display_method = (
        "Fixed residual quantile" if method == "Split conformal" else method
    )
    axis.set_title(
        f"{model} - {display_method} ({level_label(interval_frame)})"
    )
    axis.set_ylabel(y_label)
    axis.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    return fig


def level_label(interval_frame: pd.DataFrame) -> str:
    """Infer a compact nominal-level label for plot titles."""
    # Bounds are symmetric absolute-error intervals; the exact level is stored
    # in the summary, not the long frame.  The exercise currently uses 90%.
    return "90% interval"
