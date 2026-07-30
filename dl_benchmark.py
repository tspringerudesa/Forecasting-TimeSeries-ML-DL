"""Orchestration for leakage-safe manual-series deep-learning benchmarks.

Global synthetic panels are handled independently by ``global_models.ipynb``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from thesis.data_preparation import (
    MANUAL_DATASETS,
    create_train_val_test_splits,
    load_dataset,
)
from thesis.dl_evaluation import evaluate_dl_models
from thesis.model_selection import (
    export_model_selection,
    filter_metrics_to_selection,
    load_model_selection,
    merge_model_selections,
    selected_base_models,
    summarize_model_selection,
)


def _dataset_model_configs(
    base_configs: Dict[str, Dict],
    dataset_name: str,
    dataset_config: Dict,
    overrides: Dict[str, Dict[str, Dict]],
    seed: int,
) -> Tuple[Dict[str, Dict], Dict]:
    """Apply one manual dataset's context and common Auto-search budget."""
    dl_settings = dataset_config.get("dl", {})
    seasonality = int(dataset_config.get("seasonality", 1))
    input_size = int(dl_settings.get("input_size", 2 * seasonality))
    input_size_candidates = list(
        dl_settings.get("input_size_candidates", [input_size])
    )
    auto_num_samples = int(dl_settings.get("auto_num_samples", 10))
    auto_startup_trials = int(dl_settings.get("auto_tpe_startup_trials", 3))
    auto_max_steps = int(dl_settings.get("auto_max_steps", 1000))
    internal_val_size = int(dl_settings.get("val_size", seasonality))

    model_configs = copy.deepcopy(base_configs)
    dataset_overrides = overrides.get(dataset_name, {})
    for model_name, model_config in model_configs.items():
        model_config.update(dataset_overrides.get(model_name, {}))
        model_config["input_size"] = input_size
        model_config["random_seed"] = seed
        model_config["_input_size_candidates"] = input_size_candidates
        model_config["_auto"]["num_samples"] = auto_num_samples
        model_config["_auto"]["n_startup_trials"] = auto_startup_trials
        model_config["max_steps"] = auto_max_steps
        model_config.setdefault("enable_progress_bar", False)
        model_config.setdefault("enable_model_summary", False)
        model_config.setdefault("logger", False)

    timesnet_config = model_configs.get("AutoTimesNet")
    if timesnet_config is not None:
        # TimesNet converts the input-plus-horizon axis with an rFFT. Index
        # zero is not a valid period, so top_k must leave the DC bin out even
        # for the shortest context proposed by Optuna.
        minimum_total_length = min(input_size_candidates) + seasonality
        maximum_safe_top_k = max(1, minimum_total_length // 2 - 1)
        top_k_spec = timesnet_config.get("_search_space", {}).get("top_k")
        if top_k_spec and top_k_spec.get("type") == "categorical":
            safe_values = [
                int(value)
                for value in top_k_spec.get("values", [])
                if 0 < int(value) <= maximum_safe_top_k
            ]
            if not safe_values:
                safe_values = [maximum_safe_top_k]
            top_k_spec["values"] = sorted(set(safe_values))
        timesnet_config.setdefault("_adaptations", []).append(
            "top_k constrained to non-DC FFT bins for shortest dataset window"
        )

    protocol = {
        "num_samples": auto_num_samples,
        "tpe_startup_trials": auto_startup_trials,
        "max_steps": auto_max_steps,
        "input_size_candidates": input_size_candidates,
        "inner_val_size": internal_val_size,
    }
    return model_configs, protocol


def _config_record(
    model_name: str,
    parameters: Dict,
    source: str,
    dataset_config: Dict,
    protocol: Dict,
    horizon: int,
) -> Dict:
    return {
        "class": model_name,
        "module": "neuralforecast.auto",
        "parameters": copy.deepcopy(parameters),
        "source": source,
        "forecasting": {
            "freq": dataset_config["freq"],
            "horizon": int(horizon),
            "input_size_candidates": protocol["input_size_candidates"],
            "inner_val_size": protocol["inner_val_size"],
            "strategy": "direct multi-horizon",
            "training_loss": "MSE",
            "validation_loss": "MSE",
        },
    }


def _fixed_configs(
    model_names: Iterable[str],
    config_records: Dict[str, Dict],
) -> Dict[str, Dict]:
    fixed = {}
    for model_name in model_names:
        if model_name not in config_records:
            raise ValueError(f"No frozen configuration found for {model_name}")
        record = config_records[model_name]
        fixed[model_name] = copy.deepcopy(record["parameters"])
        fixed[model_name]["_source"] = record.get("source")
    return fixed


def _prepare_datasets(dataset_names):
    prepared = {}
    for dataset_name in dataset_names:
        if dataset_name not in MANUAL_DATASETS:
            raise ValueError(
                f"{dataset_name!r} is not a manual benchmark dataset. "
                "Use global_models.ipynb for synthetic panels."
            )
        frame, config = load_dataset(dataset_name, "manual")
        ts_train, ts_val, ts_test, id_col, time_col, target_col = (
            create_train_val_test_splits(frame, config, "manual")
        )
        prepared[dataset_name] = {
            "config": config,
            "type": "manual",
            "train": ts_train,
            "validation": ts_val,
            "test": ts_test,
            "id_col": id_col,
            "time_col": time_col,
            "target_col": target_col,
        }
    return prepared


def run_dl_benchmark(
    dataset_names,
    evaluation_split: str,
    base_configs: Dict[str, Dict],
    selection_path: Path,
    overrides: Dict[str, Dict[str, Dict]],
    seed: int,
):
    """Run all Auto architectures on manual validation or frozen test winners."""
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be 'validation' or 'test'")

    dataset_names = list(dataset_names)
    prepared = _prepare_datasets(dataset_names)
    protocols = {}
    configured_models = {}
    for dataset_name in dataset_names:
        configured_models[dataset_name], protocols[dataset_name] = (
            _dataset_model_configs(
                base_configs,
                dataset_name,
                prepared[dataset_name]["config"],
                overrides,
                seed,
            )
        )

    benchmark_protocol = {
        "seed": seed,
        "candidate_models": list(base_configs),
        "base_configs": copy.deepcopy(base_configs),
        "datasets": protocols,
    }

    if evaluation_split == "test":
        if not selection_path.exists():
            raise FileNotFoundError(
                f"No frozen validation selection found at {selection_path}"
            )
        frozen_selection = load_model_selection(selection_path)
        saved_protocol = frozen_selection.get("benchmark_protocol", {})
        if (
            saved_protocol.get("seed") != seed
            or saved_protocol.get("candidate_models") != list(base_configs)
            or saved_protocol.get("base_configs") != base_configs
        ):
            raise ValueError(
                "The saved DL candidate protocol differs from this test run; "
                "rerun validation first."
            )
        for dataset_name, protocol in protocols.items():
            if saved_protocol.get("datasets", {}).get(dataset_name) != protocol:
                raise ValueError(
                    f"{dataset_name}: saved DL dataset protocol differs from test"
                )
        return _run_test(prepared, frozen_selection, benchmark_protocol)

    all_results = {}
    configs_by_dataset = {}
    metrics_by_dataset = {}
    for dataset_index, dataset_name in enumerate(dataset_names, start=1):
        data = prepared[dataset_name]
        protocol = protocols[dataset_name]
        label = f"[manual dataset {dataset_index}/{len(dataset_names)}: {dataset_name}]"
        print(
            f"\n{label} {len(base_configs)} Auto searches | "
            f"{protocol['num_samples']} trials/architecture"
        )
        results, metrics, _, resolved = evaluate_dl_models(
            data["train"],
            data["validation"],
            configured_models[dataset_name],
            data["config"]["freq"],
            protocol["inner_val_size"],
            data["id_col"],
            data["time_col"],
            data["target_col"],
            tune=True,
            progress_label=label,
        )
        horizon = int(
            data["validation"].groupby(data["id_col"]).size().iloc[0]
        )
        records = {
            model_name: _config_record(
                model_name,
                parameters,
                configured_models[dataset_name][model_name].get("_source"),
                data["config"],
                protocol,
                horizon,
            )
            for model_name, parameters in resolved.items()
        }
        configs_by_dataset[dataset_name] = {"Original": records}
        metrics_by_dataset[dataset_name] = metrics
        all_results[dataset_name] = {
            **data,
            "fit_df": data["train"],
            "eval_df": data["validation"],
            "results": results,
            "metrics": metrics,
            "fitted_models": {},
            "resolved_configs": resolved,
        }

    selection_update = summarize_model_selection(
        metrics_by_dataset,
        configs_by_dataset,
        metric="rmse",
        series_col={
            name: prepared[name]["id_col"] for name in metrics_by_dataset
        },
    )
    existing = (
        load_model_selection(selection_path)
        if Path(selection_path).exists()
        else {}
    )
    selection = merge_model_selections(existing, selection_update)
    saved_protocol = dict(existing.get("benchmark_protocol", {}))
    saved_protocol.update(
        {
            "seed": seed,
            "candidate_models": list(base_configs),
            "base_configs": copy.deepcopy(base_configs),
        }
    )
    saved_protocol.setdefault("datasets", {}).update(protocols)
    selection["benchmark_protocol"] = saved_protocol
    export_model_selection(selection, selection_path)
    return all_results, configs_by_dataset, selection, benchmark_protocol


def _run_test(prepared, selection, benchmark_protocol):
    all_results = {}
    configs_by_dataset = {}
    for dataset_index, (dataset_name, data) in enumerate(
        prepared.items(), start=1
    ):
        fit_df = pd.concat(
            [data["train"], data["validation"]], ignore_index=True
        )
        eval_df = data["test"]
        selected_models = selected_base_models(
            selection, dataset_name, "Original"
        )
        series_winners = selection["per_series"][dataset_name]
        config_records = {
            winner["Original"]["base_model"]: winner["Original"]["config"]
            for winner in series_winners.values()
        }
        fixed = _fixed_configs(selected_models, config_records)
        label = f"[test dataset {dataset_index}/{len(prepared)}: {dataset_name}]"
        print(
            f"\n{label} models={selected_models} | "
            f"series={fit_df[data['id_col']].nunique()} | Optuna disabled"
        )
        results, metrics, _, resolved = evaluate_dl_models(
            fit_df,
            eval_df,
            fixed,
            data["config"]["freq"],
            0,
            data["id_col"],
            data["time_col"],
            data["target_col"],
            tune=False,
            progress_label=label,
        )
        metrics = filter_metrics_to_selection(
            metrics,
            selection,
            dataset_name,
            series_col=data["id_col"],
        )
        configs_by_dataset[dataset_name] = {
            "Original": {
                name: config_records[name] for name in selected_models
            }
        }
        all_results[dataset_name] = {
            **data,
            "fit_df": fit_df,
            "eval_df": eval_df,
            "results": results,
            "metrics": metrics,
            "fitted_models": {},
            "resolved_configs": resolved,
        }
    return all_results, configs_by_dataset, selection, benchmark_protocol
