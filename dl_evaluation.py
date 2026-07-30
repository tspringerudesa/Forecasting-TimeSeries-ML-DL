"""Leakage-safe NeuralForecast evaluation for thesis deep-learning models."""

import copy
import gc
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.ts_utils import forecast_bias_NIXTLA
from thesis.runtime import neuralforecast_compute_parameters


DL_MODEL_NAMES = (
    "AutoRNN",
    "AutoLSTM",
    "AutoGRU",
    "AutoVanillaTransformer",
    "AutoAutoformer",
    "AutoPatchTST",
    "AutoiTransformer",
    "AutoTFT",
    "AutoTimesNet",
    "AutoNLinear",
    "AutoTSMixer",
    "AutoTiDE",
    "AutoNBEATS",
    "AutoNHITS",
    "AutoKAN",
)


def _format_duration(seconds: float) -> str:
    """Format progress durations without pulling in another dependency."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def load_dl_configs(config_dir: str, model_names=DL_MODEL_NAMES) -> Dict[str, Dict]:
    """Load thesis-specific JSON configs while retaining provenance metadata."""
    config_dir = Path(config_dir)
    configs = {}
    for model_name in model_names:
        path = config_dir / f"{model_name}_config.json"
        with path.open(encoding="utf-8") as file:
            configs[model_name] = json.load(file)
        configs[model_name].update(neuralforecast_compute_parameters())
    return configs


def model_parameters(config: Dict) -> Dict:
    """Remove underscore-prefixed provenance fields before model construction."""
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _suggest_optuna_parameter(trial, name: str, specification: Dict, config: Dict):
    """Translate a JSON search-space entry into one Optuna suggestion."""
    parameter_type = specification["type"]
    if parameter_type == "dataset_candidates":
        values = config.get("_input_size_candidates")
        if not values:
            values = [model_parameters(config)["input_size"]]
        return trial.suggest_categorical(name, values)
    if parameter_type == "categorical":
        return trial.suggest_categorical(name, specification["values"])
    if parameter_type == "int":
        return trial.suggest_int(
            name,
            int(specification["low"]),
            int(specification["high"]),
            step=int(specification.get("step", 1)),
            log=bool(specification.get("log", False)),
        )
    if parameter_type == "float":
        return trial.suggest_float(
            name,
            float(specification["low"]),
            float(specification["high"]),
            step=specification.get("step"),
            log=bool(specification.get("log", False)),
        )
    raise ValueError(f"Unsupported Optuna parameter type for {name}: {parameter_type}")


def _optuna_config(config: Dict):
    """Build the callable configuration required by NeuralForecast's Optuna backend."""
    search_space = config.get("_search_space", {})
    constants = model_parameters(config)
    for parameter_name in search_space:
        constants.pop(parameter_name, None)

    def configure(trial):
        parameters = constants.copy()
        parameters.update(
            {
                name: _suggest_optuna_parameter(trial, name, specification, config)
                for name, specification in search_space.items()
            }
        )
        return parameters

    return configure


def _is_auto_config(config: Dict) -> bool:
    return bool(config.get("_auto"))


def ensure_timesnet_fft_safe(config: Dict, horizon: int) -> Dict:
    """Return a TimesNet config whose FFT top-k excludes the DC bin.

    TimesNet derives periods from an rFFT over ``input_size + horizon``.
    The zero-frequency bin cannot define a period, so an Auto search must
    request no more than the number of non-DC bins available for its shortest
    candidate context. Frozen configurations are validated rather than
    silently changed.
    """
    safe_config = copy.deepcopy(config)
    candidates = safe_config.get("_input_size_candidates")
    if candidates:
        minimum_input_size = min(int(value) for value in candidates)
    else:
        minimum_input_size = int(model_parameters(safe_config)["input_size"])
    maximum_top_k = max(1, (minimum_input_size + int(horizon)) // 2)

    top_k_spec = safe_config.get("_search_space", {}).get("top_k")
    if top_k_spec and top_k_spec.get("type") == "categorical":
        safe_values = sorted(
            {
                int(value)
                for value in top_k_spec.get("values", [])
                if 0 < int(value) <= maximum_top_k
            }
        )
        if not safe_values:
            safe_values = [maximum_top_k]
        top_k_spec["values"] = safe_values
        return safe_config

    fixed_top_k = model_parameters(safe_config).get("top_k")
    if fixed_top_k is not None and not 0 < int(fixed_top_k) <= maximum_top_k:
        raise ValueError(
            f"TimesNet top_k={fixed_top_k} is invalid for the shortest "
            f"input_size={minimum_input_size} and horizon={int(horizon)}; "
            f"the maximum FFT-safe value is {maximum_top_k}."
        )
    return safe_config


def _cleanup_optuna_trial(study, trial):
    """Release trial objects and unused CUDA cache between sequential fits."""
    del study, trial
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def build_dl_model(
    model_name: str,
    horizon: int,
    config: Dict,
    n_series: int,
    alias: Optional[str] = None,
):
    """Instantiate one Auto NeuralForecast architecture from a JSON search space."""
    from neuralforecast.losses.pytorch import MSE
    from neuralforecast.auto import (
        AutoAutoformer,
        AutoGRU,
        AutoKAN,
        AutoLSTM,
        AutoNBEATS,
        AutoNHITS,
        AutoNLinear,
        AutoPatchTST,
        AutoRNN,
        AutoTFT,
        AutoTSMixer,
        AutoTiDE,
        AutoTimesNet,
        AutoVanillaTransformer,
        AutoiTransformer,
    )

    model_classes = {
        "AutoRNN": AutoRNN,
        "AutoLSTM": AutoLSTM,
        "AutoGRU": AutoGRU,
        "AutoVanillaTransformer": AutoVanillaTransformer,
        "AutoAutoformer": AutoAutoformer,
        "AutoPatchTST": AutoPatchTST,
        "AutoiTransformer": AutoiTransformer,
        "AutoTFT": AutoTFT,
        "AutoTimesNet": AutoTimesNet,
        "AutoNLinear": AutoNLinear,
        "AutoTSMixer": AutoTSMixer,
        "AutoTiDE": AutoTiDE,
        "AutoNBEATS": AutoNBEATS,
        "AutoNHITS": AutoNHITS,
        "AutoKAN": AutoKAN,
    }
    if model_name not in model_classes:
        raise ValueError(f"Unsupported deep-learning model: {model_name}")

    if model_name == "AutoTimesNet":
        config = ensure_timesnet_fft_safe(config, horizon)

    if not _is_auto_config(config):
        raise ValueError(
            f"{model_name} must use an Auto configuration so every architecture "
            "receives the same validation-based selection procedure"
        )
    if "max_steps" in config.get("_search_space", {}):
        raise ValueError(
            "max_steps must be a fixed dataset-level ceiling, not an Optuna "
            "search dimension"
        )

    import optuna

    auto_settings = config["_auto"]
    if auto_settings.get("backend", "optuna") != "optuna":
        raise ValueError("Thesis Auto models currently require the Optuna backend")
    num_samples = int(auto_settings.get("num_samples", 10))
    startup_trials = int(
        auto_settings.get("n_startup_trials", min(3, max(1, num_samples // 2)))
    )
    if not 0 < startup_trials <= num_samples:
        raise ValueError("n_startup_trials must be between 1 and num_samples")

    model_config = config.copy()
    auto_kwargs = {}
    # These two models are multivariate in NeuralForecast 1.7.4. Their Auto
    # wrappers require n_series both at wrapper construction and as a fixed
    # value returned by the Optuna configuration callable.
    if model_name in {"AutoiTransformer", "AutoTSMixer"}:
        model_config["n_series"] = n_series
        auto_kwargs["n_series"] = n_series

    return model_classes[model_name](
        h=horizon,
        loss=MSE(),
        valid_loss=MSE(),
        config=_optuna_config(model_config),
        search_alg=optuna.samplers.TPESampler(
            seed=int(model_parameters(model_config).get("random_seed", 67)),
            n_startup_trials=startup_trials,
        ),
        num_samples=num_samples,
        refit_with_val=bool(auto_settings.get("refit_with_val", True)),
        verbose=bool(auto_settings.get("verbose", False)),
        backend="optuna",
        callbacks=[_cleanup_optuna_trial],
        alias=alias or model_name,
        **auto_kwargs,
    )


def build_fixed_dl_model(
    model_name: str,
    horizon: int,
    config: Dict,
    n_series: int,
    alias: Optional[str] = None,
):
    """Instantiate a base architecture from validation-resolved parameters."""
    from neuralforecast.losses.pytorch import MSE
    from neuralforecast.models import (
        Autoformer,
        GRU,
        KAN,
        LSTM,
        NBEATS,
        NHITS,
        NLinear,
        PatchTST,
        RNN,
        TFT,
        TSMixer,
        TiDE,
        TimesNet,
        VanillaTransformer,
        iTransformer,
    )

    model_classes = {
        "AutoRNN": RNN,
        "AutoLSTM": LSTM,
        "AutoGRU": GRU,
        "AutoVanillaTransformer": VanillaTransformer,
        "AutoAutoformer": Autoformer,
        "AutoPatchTST": PatchTST,
        "AutoiTransformer": iTransformer,
        "AutoTFT": TFT,
        "AutoTimesNet": TimesNet,
        "AutoNLinear": NLinear,
        "AutoTSMixer": TSMixer,
        "AutoTiDE": TiDE,
        "AutoNBEATS": NBEATS,
        "AutoNHITS": NHITS,
        "AutoKAN": KAN,
    }
    if model_name not in model_classes:
        raise ValueError(f"Unsupported fixed deep-learning model: {model_name}")

    if model_name == "AutoTimesNet":
        config = ensure_timesnet_fft_safe(config, horizon)
    parameters = model_parameters(config)
    parameters.pop("h", None)
    parameters.pop("alias", None)
    parameters["loss"] = MSE()
    parameters["valid_loss"] = MSE()
    # Final fitting uses every train+validation observation. Early stopping was
    # a validation-only selection device and must not reserve another holdout.
    parameters["early_stop_patience_steps"] = -1
    parameters.update(neuralforecast_compute_parameters())
    if model_name in {"AutoiTransformer", "AutoTSMixer"}:
        parameters["n_series"] = n_series

    return model_classes[model_name](
        h=horizon,
        alias=alias or model_name,
        **parameters,
    )


def _auto_fit_details(fitted_model, total_fit_time: float):
    """Return Optuna tuning time, refit time, and the winning configuration."""
    study = fitted_model.results
    tuning_time = sum(
        trial.duration.total_seconds()
        for trial in study.trials
        if trial.duration is not None
    )
    refit_time = max(0.0, total_fit_time - tuning_time)
    best_config = study.best_trial.user_attrs["ALL_PARAMS"].copy()
    for loss_name in ("loss", "valid_loss"):
        if loss_name in best_config:
            best_config[loss_name] = type(best_config[loss_name]).__name__
    return tuning_time, refit_time, best_config


def _forecast_frame(forecast: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    if id_col in forecast.columns and time_col in forecast.columns:
        return forecast.copy()
    return forecast.reset_index()


def evaluate_dl_models(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    configs: Dict[str, Dict],
    freq: str,
    internal_val_size: int,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    retain_fitted_models: bool = False,
    tune: bool = True,
    progress_label: str = "",
):
    """Fit and score Auto candidates or frozen validation-selected neural models."""
    from neuralforecast import NeuralForecast

    required = {id_col, time_col, target_col}
    for frame_name, frame in (("fit_df", fit_df), ("eval_df", eval_df)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{frame_name} missing columns: {sorted(missing)}")

    horizon_sizes = eval_df.groupby(id_col).size()
    if horizon_sizes.empty or horizon_sizes.nunique() != 1:
        raise ValueError("Every series must have the same non-empty evaluation horizon")
    horizon = int(horizon_sizes.iloc[0])
    history_sizes = fit_df.groupby(id_col).size()
    if tune and internal_val_size < 1:
        raise ValueError("Deep-learning early stopping requires a positive val_size")
    if not tune and internal_val_size != 0:
        raise ValueError("Frozen final-test fitting must use internal_val_size=0")

    input_size_values = []
    for config in configs.values():
        candidates = config.get("_input_size_candidates")
        if candidates:
            input_size_values.extend(int(value) for value in candidates)
            continue
        fixed_input_size = model_parameters(config).get("input_size")
        if fixed_input_size is None:
            raise ValueError(
                "Each DL configuration must provide either a fixed input_size "
                "or non-empty _input_size_candidates."
            )
        input_size_values.append(int(fixed_input_size))
    max_input_size = max(input_size_values)
    minimum_required = max_input_size + internal_val_size + horizon
    if int(history_sizes.min()) < minimum_required:
        raise ValueError(
            f"Shortest fitting series has {int(history_sizes.min())} rows; "
            f"at least {minimum_required} are required for input, inner validation, and horizon"
        )

    train_data = fit_df[[id_col, time_col, target_col]].copy()

    results = eval_df[[id_col, time_col, target_col]].copy()
    results[time_col] = pd.to_datetime(results[time_col])
    metric_rows = []
    fitted_models = {}
    resolved_configs = {}
    n_series = int(train_data[id_col].nunique())

    if tune:
        optimization_budgets = {
            (
                int(config["_auto"]["num_samples"]),
                int(config["_auto"].get("n_startup_trials", 0)),
                int(model_parameters(config)["max_steps"]),
                int(model_parameters(config)["val_check_steps"]),
                int(model_parameters(config)["early_stop_patience_steps"]),
            )
            for config in configs.values()
        }
        if len(optimization_budgets) != 1:
            raise ValueError(
                "All deep-learning candidates must share the same trial count, "
                "startup count, step ceiling, and early-stopping schedule"
            )

    model_count = len(configs)
    progress_start = time.perf_counter()
    phase_name = "Auto tuning" if tune else "fixed refit"
    for model_index, (model_name, config) in enumerate(configs.items(), start=1):
        prefix = f"{progress_label} | " if progress_label else ""
        print(
            f"{prefix}{phase_name}: model {model_index}/{model_count} "
            f"({model_name}) started"
        )
        model_start = time.perf_counter()
        forecast_column = model_name
        builder = build_dl_model if tune else build_fixed_dl_model
        model = builder(
            model_name, horizon, config, n_series=n_series, alias=forecast_column
        )
        neural_forecast = NeuralForecast(models=[model], freq=freq)

        training_start = time.perf_counter()
        try:
            neural_forecast.fit(
                df=train_data,
                val_size=internal_val_size,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                verbose=False,
            )
        except Exception:
            # Failed Optuna trials can otherwise leave Lightning objects and
            # CUDA allocations referenced by the notebook traceback.
            del neural_forecast
            del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            raise
        training_elapsed = time.perf_counter() - training_start
        fitted_model = neural_forecast.models[0]
        if tune:
            tuning_elapsed, final_training_elapsed, resolved_config = (
                _auto_fit_details(fitted_model, training_elapsed)
            )
        else:
            tuning_elapsed = 0.0
            final_training_elapsed = training_elapsed
            resolved_config = model_parameters(config)
        resolved_configs[model_name] = resolved_config

        forecast_start = time.perf_counter()
        forecast = neural_forecast.predict(
            futr_df=eval_df[[id_col, time_col]].copy()
        )
        forecast_elapsed = time.perf_counter() - forecast_start
        forecast = _forecast_frame(forecast, id_col, time_col)
        forecast[time_col] = pd.to_datetime(forecast[time_col])

        if forecast_column not in forecast.columns:
            prediction_columns = [
                column for column in forecast.columns if column not in {id_col, time_col}
            ]
            if len(prediction_columns) != 1:
                raise ValueError(
                    f"Could not identify prediction column for {forecast_column}: "
                    f"{prediction_columns}"
                )
            forecast = forecast.rename(columns={prediction_columns[0]: forecast_column})

        results = results.merge(
            forecast[[id_col, time_col, forecast_column]],
            on=[id_col, time_col],
            how="left",
            validate="one_to_one",
        )

        for series_id, group in results.groupby(id_col, sort=False):
            ordered = group.sort_values(time_col)
            actual = ordered[target_col].astype(float).to_numpy()
            predicted = ordered[forecast_column].astype(float).to_numpy()
            bias_frame = pd.DataFrame(
                {id_col: series_id, target_col: actual, forecast_column: predicted}
            )
            bias = forecast_bias_NIXTLA(
                bias_frame,
                models=[forecast_column],
                id_col=id_col,
                target_col=target_col,
            )[forecast_column].iloc[0]
            metric_rows.append(
                {
                    id_col: series_id,
                    "Model": forecast_column,
                    "Base Model": model_name,
                    "Target Transformation": "Original",
                    "Configuration Source": config.get("_source"),
                    "mae": float(np.mean(np.abs(actual - predicted))),
                    "mse": float(np.mean(np.square(actual - predicted))),
                    "rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
                    "forecast_bias": float(bias),
                    "Preprocessing Time": 0.0,
                    "Tuning Time": tuning_elapsed,
                    "Training Time": final_training_elapsed,
                    "Forecast Time": forecast_elapsed,
                    "Time Elapsed": (
                        training_elapsed + forecast_elapsed
                    ),
                }
            )
        if retain_fitted_models:
            fitted_models[forecast_column] = neural_forecast
        else:
            # Forecasts and the resolved configuration are sufficient for this
            # benchmark. Releasing the Lightning/Optuna objects prevents the
            # sequential searches from accumulating CPU and CUDA memory.
            del fitted_model
            del neural_forecast
            del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        model_elapsed = time.perf_counter() - model_start
        phase_elapsed = time.perf_counter() - progress_start
        mean_model_seconds = phase_elapsed / model_index
        phase_eta = mean_model_seconds * (model_count - model_index)
        print(
            f"{prefix}{phase_name}: model {model_index}/{model_count} "
            f"({model_name}) completed in {_format_duration(model_elapsed)} | "
            f"phase elapsed {_format_duration(phase_elapsed)} | "
            f"phase ETA {_format_duration(phase_eta)}"
        )

    return (
        results.sort_values([id_col, time_col]).reset_index(drop=True),
        pd.DataFrame(metric_rows),
        fitted_models,
        resolved_configs,
    )


def evaluate_replicated_dl_models(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    configs: Dict[str, Dict],
    freq,
    internal_val_size: int,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    tune: bool = True,
    progress_label: str = "",
):
    """Tune once across replications, then fit and score each replication locally.

    The panel-wide Auto search chooses one hyperparameter configuration per
    architecture from the aligned inner validation windows. Forecast metrics
    are then produced by independent fixed-configuration fits, one for each
    seeded replication.
    """
    replication_ids = sorted(set(fit_df[id_col].astype(str)))
    model_count = len(configs)
    if tune:
        print(
            f"{progress_label} | phase 1/2: panel-wide Auto search for "
            f"{model_count} architectures across {len(replication_ids)} replications"
        )
        _, tuning_metrics, _, resolved_configs = evaluate_dl_models(
            fit_df,
            eval_df,
            configs,
            freq,
            internal_val_size,
            id_col,
            time_col,
            target_col,
            retain_fitted_models=False,
            tune=True,
            progress_label=f"{progress_label} | tuning",
        )
        tuning_time_by_model = (
            tuning_metrics.groupby("Model")["Tuning Time"].first().to_dict()
        )
        fixed_configs = {
            model_name: {
                **parameters,
                "_source": configs[model_name].get("_source"),
            }
            for model_name, parameters in resolved_configs.items()
        }
    else:
        fixed_configs = configs
        resolved_configs = {
            model_name: model_parameters(config)
            for model_name, config in configs.items()
        }
        tuning_time_by_model = {model_name: 0.0 for model_name in configs}

    result_parts = []
    metric_parts = []
    fitted_models = {}
    fit_ids = set(replication_ids)
    eval_ids = set(eval_df[id_col].astype(str))
    if fit_ids != eval_ids:
        raise ValueError("Fit and evaluation frames must contain the same replications")

    replication_start = time.perf_counter()
    for replication_index, series_id in enumerate(sorted(fit_ids), start=1):
        phase_label = (
            "phase 2/2: fixed local refits"
            if tune else "fixed validation-selected evaluation"
        )
        print(
            f"{progress_label} | {phase_label} | replication "
            f"{replication_index}/{len(fit_ids)} ({series_id})"
        )
        series_fit = fit_df.loc[
            fit_df[id_col].astype(str).eq(series_id)
        ].copy()
        series_eval = eval_df.loc[
            eval_df[id_col].astype(str).eq(series_id)
        ].copy()
        results, metrics, series_models, _ = evaluate_dl_models(
            series_fit,
            series_eval,
            fixed_configs,
            freq,
            0,
            id_col,
            time_col,
            target_col,
            retain_fitted_models=False,
            tune=False,
            progress_label=(
                f"{progress_label} | replication "
                f"{replication_index}/{len(fit_ids)}"
            ),
        )
        if tune:
            metrics["Tuning Time"] = metrics["Model"].map(
                tuning_time_by_model
            ).astype(float)
            metrics["Time Elapsed"] = (
                metrics["Time Elapsed"] + metrics["Tuning Time"]
            )
        result_parts.append(results)
        metric_parts.append(metrics)
        if series_models:
            fitted_models[series_id] = series_models
        # `evaluate_dl_models` already releases every fitted Lightning model
        # after forecasting. This replication-boundary cleanup also releases
        # the local frames and any allocator cache left between local fits.
        del series_fit, series_eval, series_models
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        replication_elapsed = time.perf_counter() - replication_start
        replication_eta = (
            replication_elapsed / replication_index
        ) * (len(fit_ids) - replication_index)
        print(
            f"{progress_label} | completed replication "
            f"{replication_index}/{len(fit_ids)} | "
            f"replication phase elapsed {_format_duration(replication_elapsed)} | "
            f"replication phase ETA {_format_duration(replication_eta)}"
        )

    return (
        pd.concat(result_parts, ignore_index=True).sort_values(
            [id_col, time_col]
        ).reset_index(drop=True),
        pd.concat(metric_parts, ignore_index=True),
        fitted_models,
        resolved_configs,
    )
