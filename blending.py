"""Leakage-safe validation-trained blending for thesis forecast experiments."""

import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.utils.ts_utils import forecast_bias_NIXTLA


BLEND_MODEL_NAME = "RidgeBlend_ClassicalML"
TARGET_TRANSFORMATIONS = ("Original", "AutoStationary")


def blend_model_name(target_transformation: str) -> str:
    """Return the forecast-column name for one target representation."""
    if target_transformation == "Original":
        return BLEND_MODEL_NAME
    if target_transformation == "AutoStationary":
        return f"{BLEND_MODEL_NAME}_AutoStationary"
    raise ValueError(f"Unsupported target transformation: {target_transformation}")


def _validation_best_classical(
    classical_selection: Dict,
    dataset_name: str,
    series_id: str,
    target_transformation: str,
) -> Dict:
    """Return the classical validation winner within one target representation."""
    dataset_selections = classical_selection.get("per_series", {}).get(dataset_name)
    if not dataset_selections:
        available = sorted(classical_selection.get("per_series", {}))
        raise KeyError(
            f"No classical validation selection for {dataset_name!r}; "
            f"available datasets are {available}. Run classical validation "
            "in batch mode before fitting the blends."
        )
    requested = str(series_id)
    matching_keys = [
        key
        for key in dataset_selections
        if str(key).casefold() == requested.casefold()
    ]
    if len(matching_keys) == 1:
        selection_key = matching_keys[0]
    elif len(dataset_selections) == 1:
        # Every current manual dataset is univariate. Accept a harmless label
        # difference (for example REAL_GDP_US versus real_gdp_us) without
        # borrowing a winner from another dataset.
        selection_key = next(iter(dataset_selections))
    else:
        raise KeyError(
            f"{dataset_name}: no classical validation selection matches "
            f"series {series_id!r}; available series are "
            f"{sorted(map(str, dataset_selections))}"
        )
    target_winners = dataset_selections[selection_key]
    if target_transformation not in target_winners:
        raise KeyError(
            f"{dataset_name} / {selection_key}: no classical "
            f"{target_transformation} validation winner"
        )
    return target_winners[target_transformation]


def _matching_series_value(
    frame: pd.DataFrame,
    id_col: str,
    requested,
    *,
    frame_name: str,
):
    """Return the unique actual identifier matching a requested series."""
    values = frame[id_col].drop_duplicates().tolist()
    matches = [
        value
        for value in values
        if str(value).casefold() == str(requested).casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(values) == 1:
        return values[0]
    raise ValueError(
        f"{frame_name}: series {requested!r} does not match the available "
        f"identifiers {list(map(str, values))}"
    )


def _validation_best_ml(
    ml_metrics: pd.DataFrame,
    series_id,
    series_col: str,
    selection_metric: str,
    target_transformation: str,
    aggregate_replicates: bool = False,
) -> pd.Series:
    """Choose the best ML forecast within one target representation."""
    metric_series = ml_metrics[series_col].astype(str)
    if aggregate_replicates:
        metric_series = metric_series.str.rsplit("_", n=1).str[0]
    rows = ml_metrics.loc[
        metric_series.eq(str(series_id))
        & ml_metrics["Target Transformation"].eq(target_transformation)
    ]
    if rows.empty:
        raise ValueError(f"No ML validation forecasts found for series {series_id}")
    if aggregate_replicates:
        rows = (
            rows.groupby(
                ["Model", "Base Model", "Target Transformation"],
                as_index=False,
                dropna=False,
            )[selection_metric]
            .mean()
        )
    return rows.sort_values(selection_metric).iloc[0]


def fit_validation_ridge_blenders(
    classical_predictions: pd.DataFrame,
    ml_predictions: pd.DataFrame,
    classical_selection: Dict,
    ml_metrics: pd.DataFrame,
    dataset_name: str,
    id_col: str,
    time_col: str,
    target_col: str,
    selection_metric: str = "rmse",
    alpha: float = 1.0,
    aggregate_replicates: bool = False,
) -> Dict:
    """Fit two-input Ridge blenders per series and target representation.

    Validation targets are meta-training data. The returned blender must therefore
    be evaluated only on the subsequent untouched test split.
    """
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive")

    required = {id_col, time_col, target_col}
    for frame_name, frame in (
        ("classical_predictions", classical_predictions),
        ("ml_predictions", ml_predictions),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{frame_name} missing columns: {sorted(missing)}")

    per_series = {}
    series_ids = ml_predictions[id_col].astype(str)
    if aggregate_replicates:
        series_ids = series_ids.str.rsplit("_", n=1).str[0]
    series_ids = series_ids.drop_duplicates().tolist()
    for series_id in series_ids:
        classical_series_id = _matching_series_value(
            classical_predictions,
            id_col,
            series_id,
            frame_name="classical_predictions",
        )
        ml_series_id = _matching_series_value(
            ml_predictions,
            id_col,
            series_id,
            frame_name="ml_predictions",
        )
        per_series[str(series_id)] = {}
        for target_transformation in TARGET_TRANSFORMATIONS:
            classical_winner = _validation_best_classical(
                classical_selection,
                dataset_name,
                str(series_id),
                target_transformation,
            )
            ml_winner = _validation_best_ml(
                ml_metrics,
                series_id,
                id_col,
                selection_metric,
                target_transformation,
                aggregate_replicates=aggregate_replicates,
            )
            classical_model = classical_winner["model"]
            ml_model = ml_winner["Model"]

            for source_name, frame, model_name in (
                ("classical", classical_predictions, classical_model),
                ("ML", ml_predictions, ml_model),
            ):
                if model_name not in frame.columns:
                    raise ValueError(
                        f"{dataset_name} / {series_id} / {target_transformation}: "
                        f"{source_name} prediction column {model_name!r} is unavailable"
                    )

            classical_series = classical_predictions[id_col].astype(str)
            ml_series = ml_predictions[id_col].astype(str)
            if aggregate_replicates:
                classical_series = classical_series.str.rsplit("_", n=1).str[0]
                ml_series = ml_series.str.rsplit("_", n=1).str[0]
            classical_rows = classical_predictions.loc[
                classical_series.str.casefold().eq(
                    str(classical_series_id).casefold()
                ),
                [time_col, target_col, classical_model],
            ].copy()
            ml_rows = ml_predictions.loc[
                ml_series.str.casefold().eq(str(ml_series_id).casefold()),
                [time_col, target_col, ml_model],
            ].copy()
            classical_rows[time_col] = pd.to_datetime(classical_rows[time_col])
            ml_rows[time_col] = pd.to_datetime(ml_rows[time_col])
            meta_frame = classical_rows.merge(
                ml_rows,
                on=[time_col],
                how="inner",
                suffixes=("_classical_target", "_ml_target"),
                validate="one_to_one",
            )

            classical_target = f"{target_col}_classical_target"
            ml_target = f"{target_col}_ml_target"
            if not np.allclose(
                meta_frame[classical_target].astype(float),
                meta_frame[ml_target].astype(float),
                equal_nan=False,
            ):
                raise ValueError(
                    f"{dataset_name} / {series_id}: classical and ML "
                    "validation targets differ"
                )

            feature_frame = pd.DataFrame(
                {
                    "classical_forecast": meta_frame[classical_model].astype(float),
                    "ml_forecast": meta_frame[ml_model].astype(float),
                }
            )
            target = meta_frame[classical_target].astype(float)
            if feature_frame.isna().any().any() or target.isna().any():
                raise ValueError(
                    f"{dataset_name} / {series_id} / {target_transformation}: "
                    "validation blend data contain missing values"
                )

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_frame)
            ridge = Ridge(
                alpha=float(alpha),
                fit_intercept=True,
                solver="lsqr",
            )
            fit_start = time.perf_counter()
            ridge.fit(scaled_features, target)
            fit_elapsed = time.perf_counter() - fit_start

            scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
            coefficients = np.asarray(ridge.coef_, dtype=float) / scale
            intercept = float(
                ridge.intercept_
                - np.dot(
                    np.asarray(ridge.coef_, dtype=float),
                    scaler.mean_ / scale,
                )
            )
            per_series[str(series_id)][target_transformation] = {
                "model": blend_model_name(target_transformation),
                "alpha": float(alpha),
                "intercept": intercept,
                "coefficients": {
                    "classical_forecast": float(coefficients[0]),
                    "ml_forecast": float(coefficients[1]),
                },
                "classical_model": classical_model,
                "classical_target_transformation": target_transformation,
                "ml_model": ml_model,
                "ml_target_transformation": target_transformation,
                "selection_metric": selection_metric,
                "classical_validation_score": float(
                    classical_winner["validation_score"]
                ),
                "ml_validation_score": float(ml_winner[selection_metric]),
                "meta_training_rows": int(len(meta_frame)),
                "meta_training_split": "validation",
                "fit_seconds": fit_elapsed,
            }

    return {
        "model": BLEND_MODEL_NAME,
        "protocol": (
            "Validation forecasts and targets fit per-series Ridge weights; "
            "the untouched test split is the first performance evaluation."
        ),
        "dataset": dataset_name,
        "selection_metric": selection_metric,
        "alpha": float(alpha),
        "aggregate_replicates": bool(aggregate_replicates),
        "target_transformations": list(TARGET_TRANSFORMATIONS),
        "per_series": per_series,
    }


def apply_frozen_ridge_blenders(
    classical_predictions: pd.DataFrame,
    ml_predictions: pd.DataFrame,
    blend_config: Dict,
    dataset_name: str,
    id_col: str,
    time_col: str,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply validation-fitted Ridge weights to untouched test forecasts."""
    if blend_config.get("dataset") != dataset_name:
        raise ValueError(
            f"Blend configuration is for {blend_config.get('dataset')}, not {dataset_name}"
        )

    results = ml_predictions.copy()
    results[time_col] = pd.to_datetime(results[time_col])
    blend_columns = [
        blend_model_name(target_transformation)
        for target_transformation in TARGET_TRANSFORMATIONS
    ]
    for column in blend_columns:
        results[column] = np.nan
    metric_rows = []
    aggregate_replicates = bool(blend_config.get("aggregate_replicates", False))

    for series_id, target_configs in blend_config["per_series"].items():
        if set(target_configs) != set(TARGET_TRANSFORMATIONS):
            raise ValueError(
                f"{dataset_name} / {series_id}: blend configuration predates "
                "the separate Original/AutoStationary protocol; rerun validation"
            )
        for target_transformation, series_config in target_configs.items():
            output_name = blend_model_name(target_transformation)
            classical_model = series_config["classical_model"]
            ml_model = series_config["ml_model"]
            if classical_model not in classical_predictions.columns:
                raise ValueError(
                    f"{dataset_name} / {series_id}: missing classical test "
                    f"forecast {classical_model}"
                )
            if ml_model not in results.columns:
                raise ValueError(
                    f"{dataset_name} / {series_id}: missing ML test forecast {ml_model}"
                )

            classical_series = classical_predictions[id_col].astype(str)
            result_series = results[id_col].astype(str)
            if aggregate_replicates:
                classical_series = classical_series.str.rsplit("_", n=1).str[0]
                result_series = result_series.str.rsplit("_", n=1).str[0]
            classical_series_id = _matching_series_value(
                classical_predictions,
                id_col,
                series_id,
                frame_name="classical_predictions",
            )
            result_series_id = _matching_series_value(
                results,
                id_col,
                series_id,
                frame_name="ml_predictions",
            )
            classical_rows = classical_predictions.loc[
                classical_series.str.casefold().eq(
                    str(classical_series_id).casefold()
                ),
                [time_col, classical_model],
            ].copy()
            classical_rows[time_col] = pd.to_datetime(classical_rows[time_col])
            result_rows = results.loc[
                result_series.str.casefold().eq(str(result_series_id).casefold())
            ].copy()
            joined = result_rows.merge(
                classical_rows,
                on=[time_col],
                how="left",
                validate="one_to_one",
            ).sort_values([id_col, time_col])
            if (
                joined[classical_model].isna().any()
                or joined[ml_model].isna().any()
            ):
                raise ValueError(
                    f"{dataset_name} / {series_id} / {target_transformation}: "
                    "incomplete base forecasts for blending"
                )

            prediction_start = time.perf_counter()
            prediction = (
                float(series_config["intercept"])
                + float(
                    series_config["coefficients"]["classical_forecast"]
                )
                * joined[classical_model].astype(float).to_numpy()
                + float(series_config["coefficients"]["ml_forecast"])
                * joined[ml_model].astype(float).to_numpy()
            )
            forecast_elapsed = time.perf_counter() - prediction_start

            result_index = results.loc[
                result_series.str.casefold().eq(str(result_series_id).casefold())
            ].sort_values([id_col, time_col]).index
            results.loc[result_index, output_name] = prediction
            joined = joined.copy()
            joined[output_name] = prediction
            for replication_id, replication_rows in joined.groupby(
                id_col, sort=False
            ):
                actual = replication_rows[target_col].astype(float).to_numpy()
                replication_prediction = replication_rows[output_name].astype(
                    float
                ).to_numpy()
                bias = forecast_bias_NIXTLA(
                    replication_rows[
                        [id_col, target_col, output_name]
                    ],
                    models=[output_name],
                    id_col=id_col,
                    target_col=target_col,
                )[output_name].iloc[0]
                metric_rows.append(
                    {
                        id_col: replication_id,
                        "Model": output_name,
                        "Base Model": BLEND_MODEL_NAME,
                        "Target Transformation": target_transformation,
                        "Configuration Source": (
                            "Validation-trained Ridge blend of the best classical "
                            f"and ML {target_transformation} forecasts"
                        ),
                        "mae": float(
                            np.mean(np.abs(actual - replication_prediction))
                        ),
                        "mse": float(
                            np.mean(np.square(actual - replication_prediction))
                        ),
                        "rmse": float(
                            np.sqrt(
                                np.mean(
                                    np.square(actual - replication_prediction)
                                )
                            )
                        ),
                        "forecast_bias": float(bias),
                        "Preprocessing Time": 0.0,
                        "Tuning Time": 0.0,
                        "Training Time": float(series_config["fit_seconds"]),
                        "Forecast Time": forecast_elapsed,
                        "Best Iteration": np.nan,
                        "Time Elapsed": (
                            float(series_config["fit_seconds"]) + forecast_elapsed
                        ),
                    }
                )

    for output_name in blend_columns:
        if results[output_name].isna().any():
            missing_series = results.loc[
                results[output_name].isna(), id_col
            ].astype(str).unique()
            raise ValueError(
                f"No frozen {output_name} blender for test series: "
                f"{sorted(missing_series)}"
            )
    return results, pd.DataFrame(metric_rows)
