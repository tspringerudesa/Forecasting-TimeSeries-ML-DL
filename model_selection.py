"""Model-selection summaries and configuration persistence for thesis experiments."""

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def model_config(model) -> Dict:
    """Extract a JSON-safe constructor-style configuration from a model."""
    if hasattr(model, "get_params"):
        parameters = model.get_params(deep=False)
    else:
        parameters = {
            key: value
            for key, value in getattr(model, "__dict__", {}).items()
            if not key.endswith("_") and not key.startswith("_")
        }
    return {
        "class": model.__class__.__name__,
        "module": model.__class__.__module__,
        "parameters": _json_safe(parameters),
    }


def candidate_registry(models: Dict[str, object]) -> Dict[str, Dict]:
    """Serialize a named candidate mapping; names must identify configurations."""
    return {name: model_config(model) for name, model in models.items()}


def _overall_from_rankings(
    rankings: pd.DataFrame,
    configs_by_dataset: Dict[str, Dict[str, Dict]],
) -> Dict:
    """Compute cross-dataset candidates from stored within-dataset ranks."""
    overall_by_target = {}
    for target_version, version_rankings in rankings.groupby(
        "Target Transformation", sort=False
    ):
        coverage = version_rankings["Dataset"].nunique()
        overall = (
            version_rankings.groupby(
                ["Model", "Base Model"],
                as_index=False,
                dropna=False,
            )
            .agg(mean_rank=("rank", "mean"), datasets=("Dataset", "nunique"))
        )
        overall = overall[overall["datasets"] == coverage].sort_values(
            ["mean_rank", "Model"]
        )
        if overall.empty:
            raise ValueError(
                f"No {target_version} candidate was evaluated on every dataset"
            )

        winner = overall.iloc[0]
        overall_name = winner["Base Model"]
        overall_by_target[target_version] = {
            "model": winner["Model"],
            "base_model": overall_name,
            "target_transformation": target_version,
            "mean_rank": float(winner["mean_rank"]),
            "datasets": int(winner["datasets"]),
            "configs_by_dataset": {
                dataset_name: configs.get(target_version, configs).get(overall_name)
                for dataset_name, configs in configs_by_dataset.items()
                if dataset_name in set(version_rankings["Dataset"])
            },
        }
    return overall_by_target


def summarize_model_selection(
    metrics_by_dataset: Dict[str, pd.DataFrame],
    configs_by_dataset: Dict[str, Dict[str, Dict]],
    metric: str = "rmse",
    series_col: str = "unique_id",
    aggregate_replicates=False,
) -> Dict:
    """Select target-version winners, optionally averaging seeded replications."""
    per_dataset = {}
    per_series = {}
    ranking_parts = []

    for dataset_name, metrics in metrics_by_dataset.items():
        required = {"Model", "Base Model", "Target Transformation", metric}
        missing = required - set(metrics.columns)
        if missing:
            raise ValueError(f"{dataset_name} metrics missing columns: {sorted(missing)}")

        grouped = (
            metrics.groupby(
                ["Model", "Base Model", "Target Transformation"],
                as_index=False,
                dropna=False,
            )[metric]
            .mean()
            .sort_values(metric)
        )
        # Raw and transformed targets are separate experiments. Ranking within
        # each target version prevents one version's scale from affecting the
        # other version's overall candidate selection.
        grouped["rank"] = grouped.groupby("Target Transformation")[metric].rank(
            method="average", ascending=True
        )
        grouped["Dataset"] = dataset_name
        ranking_parts.append(grouped)

        per_dataset[dataset_name] = {}
        for target_version, version_results in grouped.groupby(
            "Target Transformation", sort=False
        ):
            winner = version_results.sort_values(metric).iloc[0]
            base_model = winner["Base Model"]
            dataset_configs = configs_by_dataset.get(dataset_name, {})
            version_configs = dataset_configs.get(target_version, dataset_configs)
            per_dataset[dataset_name][target_version] = {
                "model": winner["Model"],
                "base_model": base_model,
                "target_transformation": target_version,
                "selection_metric": metric,
                "validation_score": float(winner[metric]),
                "config": version_configs.get(base_model),
            }

        per_series[dataset_name] = {}
        dataset_series_col = (
            series_col.get(dataset_name, "unique_id")
            if isinstance(series_col, dict)
            else series_col
        )
        aggregate_dataset_replicates = (
            aggregate_replicates.get(dataset_name, False)
            if isinstance(aggregate_replicates, dict)
            else bool(aggregate_replicates)
        )
        if dataset_series_col in metrics.columns:
            series_selection_metrics = metrics.copy()
            if aggregate_dataset_replicates:
                series_selection_metrics[dataset_series_col] = (
                    series_selection_metrics[dataset_series_col]
                    .astype(str)
                    .str.rsplit("_", n=1)
                    .str[0]
                )
                series_selection_metrics = (
                    series_selection_metrics.groupby(
                        [
                            dataset_series_col,
                            "Model",
                            "Base Model",
                            "Target Transformation",
                        ],
                        as_index=False,
                        dropna=False,
                    )[metric]
                    .mean()
                )
            for series_id, series_metrics in series_selection_metrics.groupby(
                dataset_series_col, sort=False
            ):
                series_key = str(series_id)
                per_series[dataset_name][series_key] = {}
                for target_version, version_results in series_metrics.groupby(
                    "Target Transformation", sort=False
                ):
                    winner = version_results.sort_values(metric).iloc[0]
                    base_model = winner["Base Model"]
                    dataset_configs = configs_by_dataset.get(dataset_name, {})
                    version_configs = dataset_configs.get(
                        target_version, dataset_configs
                    )
                    per_series[dataset_name][series_key][target_version] = {
                        "model": winner["Model"],
                        "base_model": base_model,
                        "target_transformation": target_version,
                        "selection_metric": metric,
                        "validation_score": float(winner[metric]),
                        "config": version_configs.get(base_model),
                    }

    rankings = pd.concat(ranking_parts, ignore_index=True)
    overall_by_target = _overall_from_rankings(rankings, configs_by_dataset)

    return {
        "selection_metric": metric,
        "overall_rule": "lowest mean within-dataset rank",
        "per_dataset": per_dataset,
        "per_series": per_series,
        "overall_by_target": overall_by_target,
        "candidate_rankings": rankings.to_dict(orient="records"),
        "configs_by_dataset": _json_safe(configs_by_dataset),
    }


def merge_model_selections(existing: Dict, update: Dict) -> Dict:
    """Merge restartable single-dataset validation selections safely.

    Dataset entries in ``update`` replace their previous versions. Candidate
    ranks are retained when available so the cross-dataset summary can be
    recomputed without rerunning already completed datasets. Legacy files that
    predate stored ranks remain usable for per-dataset/test/blending decisions;
    their old cross-dataset summary is retained until every dataset has been
    exported once with the new schema.
    """
    if not existing:
        return update
    if existing.get("selection_metric") != update.get("selection_metric"):
        raise ValueError(
            "Cannot merge validation selections using different metrics"
        )

    merged = {
        "selection_metric": update["selection_metric"],
        "overall_rule": update.get(
            "overall_rule", existing.get("overall_rule")
        ),
        "per_dataset": dict(existing.get("per_dataset", {})),
        "per_series": dict(existing.get("per_series", {})),
        "overall_by_target": dict(existing.get("overall_by_target", {})),
        "candidate_rankings": list(existing.get("candidate_rankings", [])),
        "configs_by_dataset": dict(existing.get("configs_by_dataset", {})),
    }
    replaced_datasets = set(update.get("per_dataset", {}))
    merged["per_dataset"].update(update.get("per_dataset", {}))
    merged["per_series"].update(update.get("per_series", {}))
    merged["configs_by_dataset"].update(update.get("configs_by_dataset", {}))
    merged["candidate_rankings"] = [
        row
        for row in merged["candidate_rankings"]
        if row.get("Dataset") not in replaced_datasets
    ]
    merged["candidate_rankings"].extend(update.get("candidate_rankings", []))

    ranked_datasets = {
        row.get("Dataset") for row in merged["candidate_rankings"]
    }
    all_datasets = set(merged["per_dataset"])
    if merged["candidate_rankings"] and ranked_datasets == all_datasets:
        rankings = pd.DataFrame(merged["candidate_rankings"])
        merged["overall_by_target"] = _overall_from_rankings(
            rankings, merged["configs_by_dataset"]
        )
        merged["overall_scope"] = "all saved datasets"
    else:
        merged["overall_scope"] = (
            "legacy partial summary; rerun validation export once per saved "
            "dataset to rebuild cross-dataset ranks"
        )
    return merged


def export_model_selection(summary: Dict, output_file: str) -> str:
    """Persist validation-selected configurations without overwriting from test."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as file:
        json.dump(_json_safe(summary), file, indent=2, sort_keys=True)
    return output_file


def export_tuning_results(results: Dict, output_file: str) -> str:
    """Persist Optuna parameters and timings for reuse during final testing."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as file:
        json.dump(_json_safe(results), file, indent=2, sort_keys=True)
    return output_file


def load_tuning_results(input_file: str) -> Dict:
    """Load validation-only tuning results for a frozen final-test run."""
    with open(input_file) as file:
        return json.load(file)


def load_model_selection(input_file: str) -> Dict:
    """Load a validation selection without changing it during final testing."""
    with open(input_file) as file:
        return json.load(file)


def selected_base_models(
    selection: Dict,
    dataset_name: str,
    target_transformation: str,
) -> List[str]:
    """Return the unique validation-selected base models needed for final testing."""
    series_winners = selection.get("per_series", {}).get(dataset_name, {})
    selected = {
        target_winners[target_transformation]["base_model"]
        for target_winners in series_winners.values()
        if target_transformation in target_winners
    }
    if not selected:
        raise ValueError(
            f"No per-series {target_transformation} winners found for {dataset_name}"
        )
    return sorted(selected)


def filter_metrics_to_selection(
    metrics: pd.DataFrame,
    selection: Dict,
    dataset_name: str,
    series_col: str = "unique_id",
    aggregate_replicates: bool = False,
) -> pd.DataFrame:
    """Keep only each series' validation-selected forecast in final-test metrics."""
    required = {series_col, "Model", "Target Transformation"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics missing selection columns: {sorted(missing)}")

    selected_rows = []
    series_winners = selection.get("per_series", {}).get(dataset_name, {})
    for series_id, target_winners in series_winners.items():
        metric_series = metrics[series_col].astype(str)
        if aggregate_replicates:
            metric_series = metric_series.str.rsplit("_", n=1).str[0]
        for target_version, winner in target_winners.items():
            mask = (
                metric_series.eq(str(series_id))
                & metrics["Target Transformation"].eq(target_version)
                & metrics["Model"].eq(winner["model"])
            )
            rows = metrics.loc[mask].copy()
            if rows.empty:
                raise ValueError(
                    f"Missing final-test forecast for {dataset_name} / {series_id} / "
                    f"{target_version} / {winner['model']}"
                )
            rows["Validation Score"] = winner["validation_score"]
            rows["Selection Metric"] = winner["selection_metric"].upper()
            selected_rows.append(rows)

    if not selected_rows:
        raise ValueError(f"No per-series validation winners found for {dataset_name}")
    return pd.concat(selected_rows, ignore_index=True)
