"""Leakage-safe recursive evaluation for thesis tree-model experiments."""

import copy
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from src.feature_engineering.autoregressive_features import add_lags
from src.utils.ts_utils import forecast_bias_NIXTLA


def make_lagged_regression_data(y: pd.Series, lags: List[int]):
    """Convert one ordered series into a lagged regression table."""
    if not isinstance(y, pd.Series):
        raise TypeError("`y` must be a pandas Series")
    if not lags or any(lag < 1 for lag in lags):
        raise ValueError("`lags` must contain positive integers")

    lag_frame, feature_columns = add_lags(
        y.rename("target").to_frame(),
        lags=sorted(set(lags)),
        column="target",
    )
    lag_frame = lag_frame.dropna(subset=feature_columns + ["target"])
    return lag_frame[feature_columns], lag_frame["target"]


def recursive_forecast(
    model: BaseEstimator,
    history: pd.Series,
    horizon: int,
    lags: List[int],
    future_index=None,
) -> pd.Series:
    """Forecast a complete horizon without using its observed target values."""
    lags = sorted(set(lags))
    if horizon < 1:
        raise ValueError("`horizon` must be positive")
    if len(history) < max(lags):
        raise ValueError("History is shorter than the largest requested lag")

    values = history.astype(float).tolist()
    feature_columns = [f"target_lag_{lag}" for lag in lags]
    predictions = []

    for _ in range(horizon):
        row = pd.DataFrame(
            [[values[-lag] for lag in lags]],
            columns=feature_columns,
        )
        prediction = float(np.asarray(model.predict(row)).reshape(-1)[0])
        predictions.append(prediction)
        values.append(prediction)

    return pd.Series(predictions, index=future_index, name="prediction")


def _fit_with_temporal_early_stopping(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    split_at: int,
    stopping_rounds: int,
    metric: str,
):
    """Select boosting iterations chronologically, then refit all fitting rows."""
    fitted = clone(estimator)
    stopping_model = clone(estimator)
    module = stopping_model.__class__.__module__.lower()
    if module.startswith("lightgbm"):
        from lightgbm import early_stopping

        stopping_model.fit(
            X_train.iloc[:split_at],
            y_train.iloc[:split_at],
            eval_set=[(X_train.iloc[split_at:], y_train.iloc[split_at:])],
            eval_metric=metric,
            callbacks=[
                early_stopping(stopping_rounds=int(stopping_rounds), verbose=False)
            ],
        )
        best_iteration = max(1, int(stopping_model.best_iteration_))
        fitted.set_params(n_estimators=best_iteration)
    else:
        raise TypeError(
            "Temporal early stopping is configured only for LightGBM"
        )
    fitted.fit(X_train, y_train)
    return fitted, best_iteration


def _rounded_median(values: List[int]) -> int:
    """Return the median iteration, with .5 rounded upward."""
    if not values:
        raise ValueError("At least one boosting iteration is required")
    return max(1, int(np.floor(float(np.median(values)) + 0.5)))


def _transform_frozen_stationary_tail(transformer, raw_tail: pd.Series) -> pd.Series:
    """Apply a fitted stationary pipeline without refitting on the later tail.

    A Box-Cox stage can receive a negative out-of-sample value after the
    inner-fitted trend and seasonal adjustments, even when its inner-training
    input was positive. In that case, enforce Box-Cox's fixed domain boundary
    at zero. The boundary is mathematical rather than estimated from the
    stopping tail, so the fitted preprocessing decisions remain unchanged.
    """
    try:
        return transformer.transform(raw_tail)
    except ValueError as error:
        if (
            "`y` values cannot be negative" not in str(error)
            or not hasattr(transformer, "_pipeline")
        ):
            raise

    transformed = raw_tail.copy()
    for stage in transformer._pipeline:
        if stage.__class__.__name__ == "BoxCoxTransformer":
            transformed = transformed.clip(lower=0.0)
        try:
            transformed = stage.transform(transformed, freq=transformer.freq)
        except TypeError:
            transformed = stage.transform(transformed)
    return transformed


def _fit_nested_transformed_lightgbm(
    estimator: BaseEstimator,
    raw_history: pd.Series,
    lags: List[int],
    transformer_factory,
    freq: str,
    stopping_fraction: float,
    stopping_rounds: int,
    metric: str,
):
    """Select LightGBM iterations without fitting the transformer on the tail."""
    if not 0 < stopping_fraction < 1:
        raise ValueError("`stopping_fraction` must be between 0 and 1")

    stopping_size = max(1, int(np.ceil(len(raw_history) * stopping_fraction)))
    split_at = len(raw_history) - stopping_size
    if split_at <= max(lags):
        raise ValueError(
            "Not enough raw inner-history observations for transformed "
            "LightGBM early stopping"
        )

    raw_inner = raw_history.iloc[:split_at]
    raw_stopping = raw_history.iloc[split_at:]

    stopping_transformer = transformer_factory()
    transformed_inner = stopping_transformer.fit_transform(raw_inner, freq=freq)
    transformed_stopping = _transform_frozen_stationary_tail(
        stopping_transformer, raw_stopping
    )

    X_inner, y_inner = make_lagged_regression_data(transformed_inner, lags)
    transformed_through_stopping = pd.concat(
        [transformed_inner, transformed_stopping]
    )
    X_through_stopping, y_through_stopping = make_lagged_regression_data(
        transformed_through_stopping, lags
    )
    stopping_index = X_through_stopping.index.intersection(
        transformed_stopping.index
    )
    X_stopping = X_through_stopping.loc[stopping_index]
    y_stopping = y_through_stopping.loc[stopping_index]
    if X_stopping.empty:
        raise ValueError("No lagged rows remain in the internal stopping tail")

    from lightgbm import early_stopping

    stopping_model = clone(estimator)
    if not stopping_model.__class__.__module__.lower().startswith("lightgbm"):
        raise TypeError("Nested transformed early stopping requires LightGBM")
    stopping_model.fit(
        X_inner,
        y_inner,
        eval_set=[(X_stopping, y_stopping)],
        eval_metric=metric,
        callbacks=[
            early_stopping(stopping_rounds=int(stopping_rounds), verbose=False)
        ],
    )
    best_iteration = max(1, int(stopping_model.best_iteration_))

    full_transformer = transformer_factory()
    transformed_full = full_transformer.fit_transform(raw_history, freq=freq)
    X_full, y_full = make_lagged_regression_data(transformed_full, lags)
    fitted = clone(estimator)
    fitted.set_params(n_estimators=best_iteration)
    fitted.fit(X_full, y_full)
    return fitted, full_transformer, transformed_full, X_full, best_iteration


def evaluate_recursive_ml_models(
    ts_train: pd.DataFrame,
    ts_test: pd.DataFrame,
    models: Dict[str, BaseEstimator],
    lags: List[int],
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    freq: str = None,
    target_transformer_factory=None,
    target_transformation: str = "Original",
    model_suffix: str = "",
    tuning_time_by_model: Optional[Dict[str, float]] = None,
    early_stopping_rounds_by_model: Optional[Dict[str, int]] = None,
    early_stopping_fraction: float = 0.2,
    early_stopping_metric: str = "rmse",
    fixed_iterations_by_model: Optional[Dict[str, int]] = None,
):
    """Fit one cloned estimator per series and recursively evaluate a horizon."""
    required = {id_col, time_col, target_col}
    for name, frame in (("ts_train", ts_train), ("ts_test", ts_test)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    results = ts_test[[id_col, time_col, target_col]].copy()
    results[time_col] = pd.to_datetime(results[time_col])
    metric_rows = []
    fitted_models = {}
    feature_rows = []

    for series_id, train_group in ts_train.groupby(id_col, sort=False):
        test_group = ts_test.loc[ts_test[id_col] == series_id].copy()
        if test_group.empty:
            continue

        train_group = train_group.sort_values(time_col)
        test_group = test_group.sort_values(time_col)
        train_index = pd.DatetimeIndex(pd.to_datetime(train_group[time_col]))
        test_index = pd.DatetimeIndex(pd.to_datetime(test_group[time_col]))
        train_y_original = pd.Series(
            train_group[target_col].astype(float).to_numpy(),
            index=train_index,
            name=target_col,
        )

        fitted_models[series_id] = {}

        for model_name, estimator in models.items():
            transformer = None
            train_y = train_y_original
            X_train = y_train = None
            fitted = clone(estimator)
            training_start = time.perf_counter()
            early_stopping_rounds = (early_stopping_rounds_by_model or {}).get(
                model_name
            )
            fixed_iteration = (fixed_iterations_by_model or {}).get(model_name)
            best_iteration = (
                max(1, int(fixed_iteration))
                if fixed_iteration is not None else None
            )
            is_lightgbm = fitted.__class__.__module__.lower().startswith("lightgbm")

            if (
                early_stopping_rounds
                and target_transformer_factory is not None
                and is_lightgbm
                and fixed_iteration is None
            ):
                (
                    fitted,
                    transformer,
                    train_y,
                    X_train,
                    best_iteration,
                ) = _fit_nested_transformed_lightgbm(
                    estimator=estimator,
                    raw_history=train_y_original,
                    lags=lags,
                    transformer_factory=target_transformer_factory,
                    freq=freq,
                    stopping_fraction=early_stopping_fraction,
                    stopping_rounds=int(early_stopping_rounds),
                    metric=early_stopping_metric,
                )
            else:
                if target_transformer_factory is not None:
                    transformer = target_transformer_factory()
                    train_y = transformer.fit_transform(
                        train_y_original, freq=freq
                    )
                X_train, y_train = make_lagged_regression_data(train_y, lags)

            if fixed_iteration is not None:
                if not is_lightgbm:
                    raise TypeError(
                        "Frozen boosting iterations are supported only for LightGBM"
                    )
                fitted.set_params(n_estimators=best_iteration)
                fitted.fit(X_train, y_train)
            elif early_stopping_rounds and not (
                target_transformer_factory is not None and is_lightgbm
            ):
                if not 0 < early_stopping_fraction < 1:
                    raise ValueError("`early_stopping_fraction` must be between 0 and 1")
                validation_size = max(
                    1, int(np.ceil(len(X_train) * early_stopping_fraction))
                )
                split_at = len(X_train) - validation_size
                if split_at < 2:
                    raise ValueError("Not enough lagged rows for temporal early stopping")

                fitted, best_iteration = _fit_with_temporal_early_stopping(
                    estimator,
                    X_train,
                    y_train,
                    split_at,
                    int(early_stopping_rounds),
                    early_stopping_metric,
                )
            elif not (
                early_stopping_rounds
                and target_transformer_factory is not None
                and is_lightgbm
            ):
                fitted.fit(X_train, y_train)
            training_elapsed = time.perf_counter() - training_start
            forecast_start = time.perf_counter()
            prediction = recursive_forecast(
                fitted,
                history=train_y,
                horizon=len(test_group),
                lags=lags,
                future_index=test_index,
            )
            if transformer is not None:
                prediction = transformer.inverse_transform(prediction)
            forecast_elapsed = time.perf_counter() - forecast_start
            final_model_elapsed = training_elapsed + forecast_elapsed

            forecast_column = f"{model_name}{model_suffix}"
            predicted = np.asarray(prediction).reshape(-1)
            result_mask = results[id_col].eq(series_id)
            result_index = results.loc[result_mask].sort_values(time_col).index
            results.loc[result_index, forecast_column] = predicted

            actual = test_group[target_col].astype(float).to_numpy()
            bias_frame = pd.DataFrame(
                {id_col: series_id, target_col: actual, forecast_column: predicted}
            )
            bias = forecast_bias_NIXTLA(
                bias_frame,
                models=[forecast_column],
                id_col=id_col,
                target_col=target_col,
            )[forecast_column].iloc[0]
            tuning_elapsed = float((tuning_time_by_model or {}).get(model_name, 0.0))
            metric_rows.append(
                {
                    id_col: series_id,
                    "Model": forecast_column,
                    "Base Model": model_name,
                    "Target Transformation": target_transformation,
                    "mae": float(np.mean(np.abs(actual - predicted))),
                    "mse": float(np.mean(np.square(actual - predicted))),
                    "rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
                    "forecast_bias": float(bias),
                    "Tuning Time": tuning_elapsed,
                    "Training Time": training_elapsed,
                    "Forecast Time": forecast_elapsed,
                    "Best Iteration": best_iteration,
                    "Time Elapsed": tuning_elapsed + final_model_elapsed,
                }
            )

            if hasattr(fitted, "feature_importances_"):
                feature_rows.extend(
                    {
                        id_col: series_id,
                        "Model": forecast_column,
                        "Target Transformation": target_transformation,
                        "Feature": feature,
                        "Importance": float(importance),
                    }
                    for feature, importance in zip(
                        X_train.columns, fitted.feature_importances_
                    )
                )
            fitted_models[series_id][forecast_column] = {
                "model": fitted,
                "transformer": transformer,
            }

    return (
        results.sort_values([id_col, time_col]).reset_index(drop=True),
        pd.DataFrame(metric_rows),
        fitted_models,
        pd.DataFrame(feature_rows),
    )


def expanding_window_splits(
    data: pd.DataFrame,
    horizon: int,
    n_splits: int,
    time_col: str = "ds",
    min_train_periods: int = 1,
):
    """Create calendar-aligned expanding-window folds for local or panel data."""
    if horizon < 1 or n_splits < 1:
        raise ValueError("`horizon` and `n_splits` must be positive")

    ordered = data.copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col])
    times = pd.Index(ordered[time_col].drop_duplicates().sort_values())
    first_validation = len(times) - n_splits * horizon
    if first_validation < min_train_periods:
        raise ValueError(
            "Not enough time periods for the requested folds, horizon, and lags"
        )

    folds = []
    for fold_number in range(n_splits):
        validation_start = first_validation + fold_number * horizon
        train_times = times[:validation_start]
        validation_times = times[validation_start : validation_start + horizon]
        train_fold = ordered[ordered[time_col].isin(train_times)].copy()
        validation_fold = ordered[ordered[time_col].isin(validation_times)].copy()
        folds.append((train_fold, validation_fold))
    return folds


def tune_recursive_ml_models(
    ts_train: pd.DataFrame,
    estimator_factories: Dict[str, Callable[[Dict], BaseEstimator]],
    parameter_suggesters: Dict[str, Callable],
    lags: List[int],
    horizon: int,
    n_splits: int = 3,
    n_trials: int = 25,
    metric: str = "rmse",
    seed: int = 67,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    freq: str = None,
    target_transformer_factory=None,
    target_transformation: str = "Original",
    show_progress_bar: bool = True,
    early_stopping_rounds_by_model: Optional[Dict[str, int]] = None,
    early_stopping_fraction: float = 0.2,
    estimator_protocols: Optional[Dict[str, Dict]] = None,
):
    """Tune each model with TPE over leakage-safe rolling-origin forecasts."""
    import optuna

    if set(estimator_factories) != set(parameter_suggesters):
        raise ValueError("Factories and parameter suggesters must have identical keys")
    if n_trials < 1:
        raise ValueError("`n_trials` must be positive")

    folds = expanding_window_splits(
        ts_train,
        horizon=horizon,
        n_splits=n_splits,
        time_col=time_col,
        min_train_periods=max(lags) + 1,
    )
    tuned_models = {}
    tuning_times = {}
    tuning_summary = {}

    model_names = list(estimator_factories)
    for model_index, model_name in enumerate(model_names, start=1):
        factory = estimator_factories[model_name]
        suggest = parameter_suggesters[model_name]
        print(
            f"  ML tuning [{model_index}/{len(model_names)}] {model_name}: "
            f"{n_trials} trials x {len(folds)} expanding folds"
        )

        def objective(trial):
            estimator = factory(suggest(trial))
            fold_scores = []
            fold_best_iterations = []
            for train_fold, validation_fold in folds:
                evaluation = evaluate_recursive_ml_models(
                    train_fold,
                    validation_fold,
                    {model_name: estimator},
                    lags,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    freq=freq,
                    target_transformer_factory=target_transformer_factory,
                    target_transformation=target_transformation,
                    early_stopping_rounds_by_model=early_stopping_rounds_by_model,
                    early_stopping_fraction=early_stopping_fraction,
                    early_stopping_metric=metric,
                )
                fold_scores.extend(evaluation[1][metric].astype(float).tolist())
                fold_best_iterations.extend(
                    evaluation[1]["Best Iteration"].dropna().astype(int).tolist()
                )
            if not fold_scores:
                raise ValueError("No validation scores were produced")
            if fold_best_iterations:
                trial.set_user_attr(
                    "fold_best_iterations", fold_best_iterations
                )
                trial.set_user_attr(
                    "selected_boosting_iteration",
                    _rounded_median(fold_best_iterations),
                )
            return float(np.mean(fold_scores))

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=min(5, n_trials), seed=seed
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        tuning_start = time.perf_counter()
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=show_progress_bar,
        )
        tuning_elapsed = time.perf_counter() - tuning_start
        print(
            f"  ML tuning [{model_index}/{len(model_names)}] {model_name} "
            f"completed in {tuning_elapsed / 60:.1f} minutes; "
            f"best {metric}={study.best_value:.6g}"
        )

        tuned_models[model_name] = factory(study.best_params)
        selected_iteration = study.best_trial.user_attrs.get(
            "selected_boosting_iteration"
        )
        if selected_iteration is not None:
            tuned_models[model_name].set_params(
                n_estimators=int(selected_iteration)
            )
        tuning_times[model_name] = tuning_elapsed
        tuning_summary[model_name] = {
            "best_params": study.best_params,
            "best_value": float(study.best_value),
            "selection_metric": metric,
            "tuning_seconds": tuning_elapsed,
            "n_trials": n_trials,
            "n_splits": n_splits,
            "horizon": horizon,
            "sampler_seed": seed,
            "estimator_protocol": copy.deepcopy(
                (estimator_protocols or {}).get(model_name, {})
            ),
            "early_stopping_rounds": (early_stopping_rounds_by_model or {}).get(
                model_name
            ),
            "early_stopping_fraction": early_stopping_fraction,
            "early_stopping_protocol": (
                "raw_inner_fit_then_frozen_outer_refit"
                if target_transformer_factory is not None
                and (early_stopping_rounds_by_model or {}).get(model_name)
                else "lagged_temporal_tail"
            ),
            "fold_best_iterations": study.best_trial.user_attrs.get(
                "fold_best_iterations"
            ),
            "selected_boosting_iteration": selected_iteration,
            "max_estimators": (
                factory(study.best_params).get_params().get("n_estimators")
                or factory(study.best_params).get_params().get("iterations")
            ),
        }

    return tuned_models, tuning_times, tuning_summary
