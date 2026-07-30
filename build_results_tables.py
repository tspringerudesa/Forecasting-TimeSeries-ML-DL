"""Build compact, traceable LaTeX tables from the saved thesis results.

This script is deliberately independent of the forecasting environments.  It
uses only the Python standard library, reads the canonical CSV/JSON artefacts,
and never fits a model.  Run it from any directory with:

    python thesis/build_results_tables.py

The generated files are written to ``thesis/results``.
"""

import csv
import json
import math
from pathlib import Path


THESIS_DIR = Path(__file__).resolve().parent
METRICS_DIR = THESIS_DIR / "metrics"
RESULTS_DIR = THESIS_DIR / "results"
CONFIG_DIR = THESIS_DIR / "model_configs"
DATA_DIR = THESIS_DIR / "data"
PREDICTIONS_DIR = THESIS_DIR / "predictions"

DATASET_LABELS = {
    "emae": "Argentina EMAE",
    "ecai_us": "US ECAI",
    "real_gdp_argentina": "Argentina real GDP",
    "real_gdp_us": "US real GDP",
}
DATASET_ORDER = ["emae", "ecai_us", "real_gdp_argentina", "real_gdp_us"]
MONTHLY_DATASETS = ["emae", "ecai_us"]
QUARTERLY_DATASETS = ["real_gdp_argentina", "real_gdp_us"]

PROFILE_LABELS = {
    "core8_long": "Core 8, long sample",
    "full13_complete_no_nir": "Full panel, complete sample",
    "full_public_native": "Full public panel, native missing values",
}
PROFILE_ORDER = [
    "core8_long",
    "full13_complete_no_nir",
    "full_public_native",
]

DGP_LABELS = {
    "autoregressive": "AR(1)",
    "pseudo_periodic": "Pseudo-periodic",
    "sinusoidal_trend_noise": "Sinusoid, trend, and noise",
}
DGP_ORDER = ["autoregressive", "pseudo_periodic", "sinusoidal_trend_noise"]

MANUAL_BASELINE_SPECS = {
    "emae": {
        "data_file": "emae-valores-anuales-indice-base-2004-mensual.csv",
        "prediction_file": "classical_test_predictions_emae.csv",
        "time": "indice_tiempo",
        "target": "emae_original",
        "season": 12,
    },
    "ecai_us": {
        "data_file": "ecai_us_monthly.csv",
        "prediction_file": "classical_test_predictions_ecai_us.csv",
        "time": "indice_tiempo",
        "target": "ecai",
        "season": 12,
    },
    "real_gdp_argentina": {
        "data_file": "real_gdp_argentina_quarterly.csv",
        "prediction_file": "classical_test_predictions_real_gdp_argentina.csv",
        "time": "indice_tiempo",
        "target": "real_gdp",
        "season": 4,
    },
    "real_gdp_us": {
        "data_file": "real_gdp_us_quarterly.csv",
        "prediction_file": "classical_test_predictions_real_gdp_us.csv",
        "time": "indice_tiempo",
        "target": "real_gdp",
        "season": 4,
    },
}

GLOBAL_DATA_FILES = {
    "autoregressive": "synthetic_ts_global_autoregressive.csv",
    "pseudo_periodic": "synthetic_ts_global_pseudo_periodic.csv",
    "sinusoidal_trend_noise": "synthetic_ts_global_sinusoidal_trend_noise.csv",
}

MAIN_LIST_CAPTIONS = {
    "tab:manual-dl-validation-selection": (
        "Manual-series neural validation shortlist"
    ),
    "tab:manual-monthly-test": "Monthly manual-series test results",
    "tab:manual-quarterly-test": "Quarterly manual-series test results",
    "tab:tree-target-transformation": (
        "Tree-pipeline results by target representation"
    ),
    "tab:global-validation-selection": "Global-model validation selection",
    "tab:global-test-selected": "Global-model test results",
    "tab:sp500-test-intervals": r"S\&P 500 interval results",
    "tab:predictable-test-intervals": "Predictable-series interval results",
    "tab:exogenous-validation-selection": (
        "Argentine inflation validation selection"
    ),
    "tab:exogenous-test-selected": "Argentine inflation test results",
    "tab:exogenous-shap-top": (
        "Grouped TreeSHAP importance by inflation-data profile"
    ),
}

DL_MODELS = {
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
}


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def number(row, column):
    value = row.get(column, "")
    if value in ("", None):
        return None
    return float(value)


def one(rows, **criteria):
    matches = [
        row
        for row in rows
        if all(str(row.get(column)) == str(value) for column, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one row for "
            + ", ".join(f"{key}={value!r}" for key, value in criteria.items())
            + f"; found {len(matches)}"
        )
    return matches[0]


def tex(value):
    text = str(value)
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def model_label(model):
    labels = {
        "Global_XGBRandomForest": "Global XGBRandomForest",
        "Global_XGBRandomForest_AutoStationary": (
            "Global XGBRandomForest (AutoStationary)"
        ),
        "RidgeBlend_ClassicalML": "Ridge blend (Original)",
        "RidgeBlend_ClassicalML_AutoStationary": (
            "Ridge blend (AutoStationary)"
        ),
        "AutoARIMA_AutoStationary": "AutoARIMA (AutoStationary)",
        "LightGBM_AutoStationary": "LightGBM (AutoStationary)",
        "XGBRandomForest_AutoStationary": (
            "XGBRandomForest (AutoStationary)"
        ),
        "AutoiTransformer": "Auto-iTransformer",
        "ZeroReturn": "Zero-return benchmark",
    }
    if model in labels:
        return labels[model]
    if model.endswith("_AutoStationary"):
        return model[: -len("_AutoStationary")] + " (AutoStationary)"
    return model


def interval_method_label(method):
    """Use a defensible display label without altering persisted result files."""
    if method == "Split conformal":
        return "Fixed residual quantile"
    return method


def fmt(value, decimals=3, scientific=False):
    if value is None:
        return "--"
    if scientific:
        return f"{value:.3e}"
    return f"{value:.{decimals}f}"


def highlight(text, green=False, bold=False):
    if bold:
        text = r"\textbf{" + text + "}"
    if green:
        text = r"\textcolor{metricgreen}{" + text + "}"
    return text


def is_min(value, values):
    present = [candidate for candidate in values if candidate is not None]
    return value is not None and present and value == min(present)


def scale_manual(dataset, value):
    if value is None:
        return None
    return value / 1000.0 if dataset == "real_gdp_argentina" else value


def table(
    caption,
    label,
    headers,
    aligns,
    rows,
    note=None,
    size=r"\small",
):
    short_caption = MAIN_LIST_CAPTIONS.get(label)
    caption_command = (
        rf"\caption[{short_caption}]{{{caption}}}"
        if short_caption
        else rf"\caption{{{caption}}}"
    )
    lines = [
        "% Generated by thesis/build_results_tables.py; do not edit by hand.",
        r"\begin{table}[H]",
        r"\centering",
        size,
        caption_command,
        r"\label{" + label + "}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{" + aligns + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
        ]
    )
    if note:
        lines.extend(
            [
                r"\par\smallskip",
                r"\begin{minipage}{0.96\textwidth}",
                r"\footnotesize\textit{Note:} " + note,
                r"\end{minipage}",
            ]
        )
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write(name, parts):
    path = RESULTS_DIR / name
    body = "\n".join(parts)
    path.write_text(body, encoding="utf-8", newline="\n")
    print(path.relative_to(THESIS_DIR.parent))


def dgp_key(row):
    value = row.get("DGP", "")
    low = value.lower()
    if value in DGP_ORDER:
        return value
    if "pseudo" in low:
        return "pseudo_periodic"
    if "sinusoid" in low:
        return "sinusoidal_trend_noise"
    if "ar(1)" in low or "autoregressive" in low:
        return "autoregressive"
    raise ValueError(f"Unrecognized DGP value: {value!r}")


def mean_absolute_error(actual, forecast):
    return sum(abs(y - y_hat) for y, y_hat in zip(actual, forecast)) / len(actual)


def root_mean_squared_error(actual, forecast):
    return math.sqrt(
        sum((y - y_hat) ** 2 for y, y_hat in zip(actual, forecast))
        / len(actual)
    )


def manual_seasonal_naive(dataset):
    """Compute a test-only seasonal-naive reference from saved source data."""
    spec = MANUAL_BASELINE_SPECS[dataset]
    source = read_csv(DATA_DIR / spec["data_file"])
    prediction_rows = read_csv(PREDICTIONS_DIR / spec["prediction_file"])
    test_dates = [row["ds"][:10] for row in prediction_rows]

    dated_values = [
        (row[spec["time"]][:10], float(row[spec["target"]]))
        for row in source
        if row.get(spec["target"], "") not in ("", None)
    ]
    dates = [item[0] for item in dated_values]
    values = [item[1] for item in dated_values]
    position = {date: index for index, date in enumerate(dates)}
    actual = []
    forecast = []
    for date in test_dates:
        index = position[date]
        if index < spec["season"]:
            raise ValueError(f"Insufficient seasonal history for {dataset}/{date}")
        actual.append(values[index])
        forecast.append(values[index - spec["season"]])

    return {
        "family": "Benchmark",
        "model": "SeasonalNaive",
        "representation": "Original",
        "mae": scale_manual(dataset, mean_absolute_error(actual, forecast)),
        "rmse": scale_manual(dataset, root_mean_squared_error(actual, forecast)),
        "time": None,
    }


def global_naive_baseline(dgp, horizon=10):
    """Compute a no-change test reference, equally averaged across series."""
    rows = read_csv(DATA_DIR / GLOBAL_DATA_FILES[dgp])
    by_series = {}
    for row in rows:
        by_series.setdefault(row["series_id"], []).append(
            (float(row["time"]), float(row["value"]))
        )
    series_mae = []
    series_rmse = []
    for series_id, observations in by_series.items():
        values = [value for _, value in sorted(observations)]
        if len(values) <= horizon:
            raise ValueError(f"Insufficient history for {dgp}/{series_id}")
        actual = values[-horizon:]
        forecast = [values[-horizon - 1]] * horizon
        series_mae.append(mean_absolute_error(actual, forecast))
        series_rmse.append(root_mean_squared_error(actual, forecast))
    return {
        "family": "Benchmark",
        "model": "Naive",
        "representation": "Original",
        "test_mae": sum(series_mae) / len(series_mae),
        "test_rmse": sum(series_rmse) / len(series_rmse),
        "test_time": None,
    }


def manual_selection_rows():
    classical_test = read_csv(METRICS_DIR / "classical_models_test_metrics_all.csv")
    ml_test = read_csv(METRICS_DIR / "ml_models_test_metrics_all.csv")
    dl_test = read_csv(METRICS_DIR / "dl_models_test_metrics_all.csv")
    classical_config = read_json(
        CONFIG_DIR / "classical_validation_selection.json"
    )
    ml_config = read_json(CONFIG_DIR / "ml_validation_selection.json")
    dl_config = read_json(CONFIG_DIR / "dl_validation_selection.json")

    output = {}
    for dataset in DATASET_ORDER:
        rows = []
        for family, config, source in (
            ("Classical", classical_config, classical_test),
            ("Machine learning", ml_config, ml_test),
        ):
            for representation in ("Original", "AutoStationary"):
                selected = config["per_dataset"][dataset][representation]
                row = one(
                    source,
                    Dataset=dataset,
                    Model=selected["model"],
                    **{"Target Transformation": representation, "Split": "test"},
                )
                rows.append(
                    {
                        "family": family,
                        "model": selected["model"],
                        "representation": representation,
                        "mae": scale_manual(dataset, number(row, "MAE")),
                        "rmse": scale_manual(dataset, number(row, "RMSE")),
                        "time": number(row, "Time Elapsed"),
                    }
                )
        selected_dl = dl_config["per_dataset"][dataset]["Original"]
        row = one(
            dl_test,
            Dataset=dataset,
            Model=selected_dl["model"],
            Split="test",
        )
        rows.append(
            {
                "family": "Deep learning",
                "model": selected_dl["model"],
                "representation": "Original",
                "mae": scale_manual(dataset, number(row, "MAE")),
                "rmse": scale_manual(dataset, number(row, "RMSE")),
                "time": number(row, "Time Elapsed"),
            }
        )
        rows.append(manual_seasonal_naive(dataset))
        output[dataset] = rows
    return output


def build_manual_test_table(datasets, frequency, label):
    selected = manual_selection_rows()
    body = []
    for dataset in datasets:
        values = selected[dataset]
        maes = [row["mae"] for row in values]
        rmses = [row["rmse"] for row in values]
        times = [row["time"] for row in values if row["time"] is not None]
        for index, row in enumerate(values):
            body.append(
                [
                    tex(DATASET_LABELS[dataset]) if index == 0 else "",
                    tex(row["family"]),
                    tex(model_label(row["model"])),
                    tex(row["representation"]),
                    highlight(fmt(row["mae"]), green=is_min(row["mae"], maes)),
                    highlight(
                        fmt(row["rmse"]),
                        green=is_min(row["rmse"], rmses),
                        bold=is_min(row["rmse"], rmses),
                    ),
                    (
                        "--"
                        if row["time"] is None
                        else highlight(
                            fmt(row["time"], 2),
                            green=is_min(row["time"], times),
                            bold=is_min(row["time"], times),
                        )
                    ),
                ]
            )
    if frequency == "monthly":
        units = (
            "Errors are in index points (EMAE: 2004=100; ECAI: 2007=100)."
        )
    else:
        units = (
            "Errors are in billions of domestic currency for Argentina and "
            "billions of chained 2017 dollars for the United States."
        )
    return table(
        (
            f"Final {frequency} manual-series test results for specifications "
            "chosen on validation."
        ),
        label,
        [
            "Series",
            "Family",
            "Model",
            "Target",
            "MAE",
            "RMSE",
            "Test time (s)",
        ],
        "llllrrr",
        body,
        (
            units
            + " Classical and machine-learning models are the validation "
            "winner within each target representation; the deep-learning model "
            "is the validation winner across the 15 architectures. "
            "SeasonalNaive is a test-only reference and does not "
            "participate in selection. Green marks "
            "the smallest displayed test value and does not alter the "
            "validation-based selection. Times are fixed-configuration test "
            "refit plus forecast times; validation tuning is excluded."
        ),
    )


def build_manual_dl_top_two():
    rows = read_csv(METRICS_DIR / "dl_models_validation_metrics_all.csv")
    body = []
    for dataset in DATASET_ORDER:
        candidates = sorted(
            [row for row in rows if row["Dataset"] == dataset],
            key=lambda row: number(row, "RMSE"),
        )[:2]
        maes = [scale_manual(dataset, number(row, "MAE")) for row in candidates]
        rmses = [scale_manual(dataset, number(row, "RMSE")) for row in candidates]
        times = [number(row, "Time Elapsed") for row in candidates]
        for rank, row in enumerate(candidates, 1):
            mae = scale_manual(dataset, number(row, "MAE"))
            rmse = scale_manual(dataset, number(row, "RMSE"))
            elapsed = number(row, "Time Elapsed")
            body.append(
                [
                    tex(DATASET_LABELS[dataset]) if rank == 1 else "",
                    str(rank),
                    tex(model_label(row["Model"])),
                    highlight(fmt(mae), green=is_min(mae, maes)),
                    highlight(
                        fmt(rmse),
                        green=is_min(rmse, rmses),
                        bold=rank == 1,
                    ),
                    highlight(
                        fmt(elapsed, 2),
                        green=is_min(elapsed, times),
                        bold=is_min(elapsed, times),
                    ),
                    r"\checkmark" if rank == 1 else "",
                ]
            )
    return table(
        (
            "Two best deep-learning architectures on each manual-series "
            "validation block."
        ),
        "tab:manual-dl-validation-selection",
        ["Series", "Rank", "Architecture", "MAE", "RMSE", "Time (s)", "Selected"],
        "lr lrrrc".replace(" ", ""),
        body,
        (
            "Architectures are ranked by validation RMSE. Each value is from "
            "one external validation block; the automatic model search is "
            "internal to that block. Argentina GDP errors are reported in "
            "billions. Time includes the reported internal tuning, fit, and "
            "validation forecast time."
        ),
    )


def build_tree_transformation():
    validation = read_csv(METRICS_DIR / "ml_models_validation_metrics_all.csv")
    test = read_csv(METRICS_DIR / "ml_models_test_metrics_all.csv")
    config = read_json(CONFIG_DIR / "ml_validation_selection.json")
    body = []
    for dataset in DATASET_ORDER:
        rows = []
        for representation in ("Original", "AutoStationary"):
            selected = config["per_dataset"][dataset][representation]
            val_row = one(
                validation,
                Dataset=dataset,
                Model=selected["model"],
                **{
                    "Target Transformation": representation,
                    "Split": "validation",
                },
            )
            test_row = one(
                test,
                Dataset=dataset,
                Model=selected["model"],
                **{"Target Transformation": representation, "Split": "test"},
            )
            rows.append(
                {
                    "representation": representation,
                    "model": selected["model"],
                    "validation": scale_manual(
                        dataset, number(val_row, "RMSE")
                    ),
                    "test": scale_manual(dataset, number(test_row, "RMSE")),
                }
            )
        val_values = [row["validation"] for row in rows]
        test_values = [row["test"] for row in rows]
        for index, row in enumerate(rows):
            body.append(
                [
                    tex(DATASET_LABELS[dataset]) if index == 0 else "",
                    tex(row["representation"]),
                    tex(model_label(row["model"])),
                    highlight(
                        fmt(row["validation"]),
                        green=is_min(row["validation"], val_values),
                    ),
                    highlight(
                        fmt(row["test"]),
                        green=is_min(row["test"], test_values),
                        bold=is_min(row["test"], test_values),
                    ),
                ]
            )
    return table(
        (
            "Validation and test RMSE of separately tuned tree pipelines by "
            "target representation."
        ),
        "tab:tree-target-transformation",
        ["Series", "Target", "Validation-selected tree", "Validation RMSE", "Test RMSE"],
        "lllrr",
        body,
        (
            "Selection is performed separately within each representation. "
            "The algorithm and its hyperparameters can therefore differ; this "
            "is not a matched transformation ablation. "
            "Test errors are evaluated after inverse transformation. Argentina "
            "GDP errors are in billions; other units follow the source series. "
            "Green compares representations within the same series and split."
        ),
    )


def build_appendix_manual():
    validation = read_csv(METRICS_DIR / "dl_models_validation_metrics_all.csv")
    dl_config = read_json(CONFIG_DIR / "dl_validation_selection.json")
    parts = []
    for dataset in DATASET_ORDER:
        candidates = sorted(
            [row for row in validation if row["Dataset"] == dataset],
            key=lambda row: number(row, "RMSE"),
        )
        selected = dl_config["per_dataset"][dataset]["Original"]["model"]
        maes = [scale_manual(dataset, number(row, "MAE")) for row in candidates]
        rmses = [scale_manual(dataset, number(row, "RMSE")) for row in candidates]
        times = [number(row, "Time Elapsed") for row in candidates]
        body = []
        for rank, row in enumerate(candidates, 1):
            mae = scale_manual(dataset, number(row, "MAE"))
            rmse = scale_manual(dataset, number(row, "RMSE"))
            elapsed = number(row, "Time Elapsed")
            body.append(
                [
                    str(rank),
                    tex(model_label(row["Model"])),
                    highlight(fmt(mae), green=is_min(mae, maes)),
                    highlight(
                        fmt(rmse),
                        green=is_min(rmse, rmses),
                        bold=row["Model"] == selected,
                    ),
                    highlight(
                        fmt(elapsed, 2),
                        green=is_min(elapsed, times),
                        bold=is_min(elapsed, times),
                    ),
                    r"\checkmark" if row["Model"] == selected else "",
                ]
            )
        parts.append(
            table(
                (
                    f"Complete deep-learning validation ranking for "
                    f"{tex(DATASET_LABELS[dataset])}."
                ),
                f"tab:appendix-dl-validation-{dataset.replace('_', '-')}",
                ["Rank", "Architecture", "MAE", "RMSE", "Time (s)", "Selected"],
                "rlrrrc",
                body,
                (
                    "Rank and selection use validation RMSE. "
                    + (
                        "Errors are divided by 1,000 and reported in billions."
                        if dataset == "real_gdp_argentina"
                        else "Errors use the source-series units."
                    )
                ),
                size=r"\scriptsize",
            )
        )

    ridge = read_csv(METRICS_DIR / "ml_models_test_metrics_all.csv")
    ridge = [row for row in ridge if row["Model Family"] == "Ensemble"]
    ridge.sort(
        key=lambda row: (
            DATASET_ORDER.index(row["Dataset"]),
            row["Target Transformation"],
        )
    )
    body = []
    for row in ridge:
        dataset = row["Dataset"]
        body.append(
            [
                tex(DATASET_LABELS[dataset]),
                tex(row["Target Transformation"]),
                tex(model_label(row["Model"])),
                fmt(scale_manual(dataset, number(row, "MAE"))),
                fmt(scale_manual(dataset, number(row, "RMSE"))),
                "--",
            ]
        )
    parts.append(
        table(
            "Validation-trained Ridge blends on the final manual-series test blocks.",
            "tab:appendix-ridge-blend-test",
            ["Series", "Target", "Ensemble", "MAE", "RMSE", "Comparable time"],
            "lllrrc",
            body,
            (
                "The two component forecasts were selected within the same "
                "target representation on validation. Comparable elapsed time "
                "is unavailable because the recorded Ridge time excludes the "
                "cost of producing its component forecasts. Argentina GDP "
                "errors are reported in billions."
            ),
            size=r"\scriptsize",
        )
    )
    return parts


def global_sources():
    validation = read_csv(METRICS_DIR / "global_models_metrics_validation.csv")
    test = read_csv(METRICS_DIR / "global_models_metrics_test.csv")
    test_timing = read_csv(METRICS_DIR / "global_models_timing_test.csv")
    dl_config = read_json(CONFIG_DIR / "global_dl_validation_configs.json")
    return validation, test, test_timing, dl_config


def global_selected():
    validation, test, test_timing, dl_config = global_sources()
    output = {}
    for dgp in DGP_ORDER:
        validation_ml = [
            row
            for row in validation
            if dgp_key(row) == dgp and row["Model Family"] == "ML"
        ]
        if len(validation_ml) != 1:
            raise ValueError(f"Expected one global ML validation row for {dgp}")
        dl_models = dl_config["per_dataset"][dgp]
        dl_model, dl_entry = min(
            dl_models.items(), key=lambda item: float(item[1]["validation_rmse"])
        )
        ml_model = validation_ml[0]["Model"]
        selected = [
            {
                "family": "Machine learning",
                "model": ml_model,
                "representation": validation_ml[0]["Target Transformation"],
                "validation_rmse": number(validation_ml[0], "Mean RMSE"),
            },
            {
                "family": "Deep learning",
                "model": dl_model,
                "representation": dl_entry["target_transformation"],
                "validation_rmse": float(dl_entry["validation_rmse"]),
            },
        ]
        for item in selected:
            test_row = [
                row
                for row in test
                if dgp_key(row) == dgp and row["Model"] == item["model"]
            ]
            if len(test_row) != 1:
                raise ValueError(
                    f"Expected one global test row for {dgp}/{item['model']}"
                )
            timing_row = [
                row
                for row in test_timing
                if dgp_key(row) == dgp and row["Model"] == item["model"]
            ]
            if len(timing_row) != 1:
                raise ValueError(
                    f"Expected one global timing row for {dgp}/{item['model']}"
                )
            item.update(
                {
                    "test_mae": number(test_row[0], "Mean MAE"),
                    "test_rmse": number(test_row[0], "Mean RMSE"),
                    "test_time": number(timing_row[0], "Time Elapsed"),
                }
            )
        output[dgp] = selected
    return output


def build_global_main():
    selected = global_selected()
    validation_body = []
    test_body = []
    for dgp in DGP_ORDER:
        rows = selected[dgp]
        test_rows = rows + [global_naive_baseline(dgp)]
        val_values = [row["validation_rmse"] for row in rows]
        test_maes = [row["test_mae"] for row in test_rows]
        test_rmses = [row["test_rmse"] for row in test_rows]
        test_times = [
            row["test_time"]
            for row in test_rows
            if row["test_time"] is not None
        ]
        for index, row in enumerate(rows):
            validation_body.append(
                [
                    tex(DGP_LABELS[dgp]) if index == 0 else "",
                    tex(row["family"]),
                    tex(model_label(row["model"])),
                    tex(row["representation"]),
                    highlight(
                        fmt(row["validation_rmse"], 4),
                        green=is_min(row["validation_rmse"], val_values),
                        bold=is_min(row["validation_rmse"], val_values),
                    ),
                ]
            )
        for index, row in enumerate(test_rows):
            test_body.append(
                [
                    tex(DGP_LABELS[dgp]) if index == 0 else "",
                    tex(row["family"]),
                    tex(model_label(row["model"])),
                    tex(row["representation"]),
                    highlight(
                        fmt(row["test_mae"], 4),
                        green=is_min(row["test_mae"], test_maes),
                    ),
                    highlight(
                        fmt(row["test_rmse"], 4),
                        green=is_min(row["test_rmse"], test_rmses),
                        bold=is_min(row["test_rmse"], test_rmses),
                    ),
                    (
                        "--"
                        if row["test_time"] is None
                        else highlight(
                            fmt(row["test_time"], 2),
                            green=is_min(row["test_time"], test_times),
                            bold=is_min(row["test_time"], test_times),
                        )
                    ),
                ]
            )
    validation_table = table(
        (
            "Validation selection for the pooled global forecasts, averaged "
            "equally across 100 series."
        ),
        "tab:global-validation-selection",
        ["DGP", "Family", "Model", "Target", "Mean RMSE"],
        "llllr",
        validation_body,
        (
            "The machine-learning row is the prespecified pooled "
            "XGBRandomForest. The deep-learning row is the architecture with "
            "the smallest mean validation RMSE in the saved configuration. "
            "The latter is authoritative when the rounded notebook CSV differs."
        ),
    )
    test_table = table(
        (
            "Final global-model test performance for the validation-selected "
            "specifications, averaged equally across 100 series."
        ),
        "tab:global-test-selected",
        ["DGP", "Family", "Model", "Target", "Mean MAE", "Mean RMSE", "Time (s)"],
        "llllrrr",
        test_body,
        (
            "Each test path contains ten simulated steps. Green compares the "
            "validation-selected specifications and the no-change "
            "benchmark within a DGP; it does not represent test-set selection. "
            "The benchmark does not receive a comparable fitted-model time. "
            "Timing is descriptive because the "
            "tree model used the CPU and deep-learning models used the GPU."
        ),
    )
    return [validation_table, test_table]


def build_appendix_global():
    validation, test, timing, dl_config = global_sources()
    parts = []
    for dgp in DGP_ORDER:
        ml_rows = [
            row
            for row in validation
            if dgp_key(row) == dgp and row["Model Family"] == "ML"
        ]
        if len(ml_rows) != 1:
            raise ValueError(f"Expected one ML validation row for {dgp}")
        candidates = [
            {
                "family": "Machine learning",
                "model": ml_rows[0]["Model"],
                "representation": ml_rows[0]["Target Transformation"],
                "rmse": number(ml_rows[0], "Mean RMSE"),
                "selected": True,
            }
        ]
        dl_entries = dl_config["per_dataset"][dgp]
        best_dl = min(
            dl_entries.items(), key=lambda item: float(item[1]["validation_rmse"])
        )[0]
        for model, entry in dl_entries.items():
            candidates.append(
                {
                    "family": entry.get("dl_category", "Deep learning"),
                    "model": model,
                    "representation": entry["target_transformation"],
                    "rmse": float(entry["validation_rmse"]),
                    "selected": model == best_dl,
                }
            )
        candidates.sort(key=lambda row: row["rmse"])
        body = []
        min_rmse = min(row["rmse"] for row in candidates)
        for rank, row in enumerate(candidates, 1):
            body.append(
                [
                    str(rank),
                    tex(row["family"]),
                    tex(model_label(row["model"])),
                    tex(row["representation"]),
                    highlight(
                        fmt(row["rmse"], 4),
                        green=row["rmse"] == min_rmse,
                        bold=row["selected"],
                    ),
                    r"\checkmark" if row["selected"] else "",
                ]
            )
        parts.append(
            table(
                (
                    f"Complete global validation ranking for the "
                    f"{tex(DGP_LABELS[dgp])} panel."
                ),
                f"tab:appendix-global-validation-{dgp.replace('_', '-')}",
                ["Rank", "Family/category", "Model", "Target", "Mean RMSE", "Retained"],
                "rlllrc",
                body,
                (
                    "Means give equal weight to each of the 100 series. The "
                    "pooled tree and the best deep-learning architecture were "
                    "retained; this is not an across-family winner-take-all "
                    "rule. Deep-learning RMSE values come from the saved "
                    "validation configuration."
                ),
                size=r"\scriptsize",
            )
        )

        test_rows = [row for row in test if dgp_key(row) == dgp]
        test_rows.sort(key=lambda row: number(row, "Mean RMSE"))
        timing_map = {
            row["Model"]: row
            for row in timing
            if dgp_key(row) == dgp
        }
        maes = [number(row, "Mean MAE") for row in test_rows]
        rmses = [number(row, "Mean RMSE") for row in test_rows]
        times = [
            number(timing_map[row["Model"]], "Time Elapsed") for row in test_rows
        ]
        body = []
        for rank, row in enumerate(test_rows, 1):
            elapsed = number(timing_map[row["Model"]], "Time Elapsed")
            mae = number(row, "Mean MAE")
            rmse = number(row, "Mean RMSE")
            body.append(
                [
                    str(rank),
                    tex(row["Model Family"]),
                    tex(model_label(row["Model"])),
                    tex(row["Target Transformation"]),
                    highlight(fmt(mae, 4), green=is_min(mae, maes)),
                    highlight(
                        fmt(rmse, 4),
                        green=is_min(rmse, rmses),
                        bold=is_min(rmse, rmses),
                    ),
                    highlight(
                        fmt(elapsed, 2),
                        green=is_min(elapsed, times),
                        bold=is_min(elapsed, times),
                    ),
                ]
            )
        parts.append(
            table(
                (
                    f"Complete global test ranking for the "
                    f"{tex(DGP_LABELS[dgp])} panel."
                ),
                f"tab:appendix-global-test-{dgp.replace('_', '-')}",
                ["Rank", "Family", "Model", "Target", "Mean MAE", "Mean RMSE", "Time (s)"],
                "rlllrrr",
                body,
                (
                    "Ranks are descriptive test rankings, not model-selection "
                    "decisions. Means give equal weight to each of 100 series. "
                    "CPU and GPU timings should not be read as a controlled "
                    "hardware comparison."
                ),
                size=r"\scriptsize",
            )
        )
    return parts


def build_sp500_main():
    rows = read_csv(RESULTS_DIR / "sp500_probabilistic_metrics_2025.csv")
    model_order = ["LightGBM", "AutoLSTM", "ZeroReturn"]
    method_order = [
        "Split conformal",
        "Online fixed-alpha",
        "Adaptive conformal inference",
    ]
    body = []
    for model in model_order:
        candidates = [row for row in rows if row["Model"] == model]
        available_methods = {row["Method"] for row in candidates}
        if not {"Split conformal", "Adaptive conformal inference"}.issubset(
            available_methods
        ):
            raise ValueError(
                f"Missing required S&P interval rows for {model}: "
                f"{sorted(available_methods)}"
            )
        candidates.sort(key=lambda row: method_order.index(row["Method"]))
        errors = [number(row, "Coverage Error") for row in candidates]
        widths = [number(row, "Mean Interval Width") for row in candidates]
        scores = [number(row, "Mean Interval Score") for row in candidates]
        for index, row in enumerate(candidates):
            error = number(row, "Coverage Error")
            width = number(row, "Mean Interval Width")
            score = number(row, "Mean Interval Score")
            body.append(
                [
                    tex(model_label(model)) if index == 0 else "",
                    tex(interval_method_label(row["Method"])),
                    fmt(number(row, "Empirical Coverage"), 3),
                    highlight(fmt(error, 3), green=is_min(error, errors)),
                    highlight(fmt(width, 5), green=is_min(width, widths)),
                    highlight(
                        fmt(score, 5),
                        green=is_min(score, scores),
                        bold=is_min(score, scores),
                    ),
                    fmt(number(row, "RMSE"), 6),
                ]
            )
    return table(
        "S\\&P 500 daily-return prediction intervals on the 2025 test sample.",
        "tab:sp500-test-intervals",
        [
            "Point model",
            "Interval method",
            "Coverage",
            "Coverage error",
            "Mean width",
            "Interval score",
            "RMSE",
        ],
        "llrrrrr",
        body,
        (
            "Nominal coverage is 0.90. Coverage error, interval width, interval "
            "score, and RMSE are lower when better, although width should be "
            "read jointly with coverage. Point models were selected on 2023, "
            "intervals calibrated on 2024, and all displayed metrics evaluated "
            "on 2025. The fixed residual-quantile benchmark reuses calibration "
            "residuals after refitting the point model and is therefore "
            "descriptive rather than an ordinary split-conformal guarantee. "
            "Online fixed-alpha appends test residuals sequentially while "
            "holding alpha at 0.10; ACI uses the same expanding history and "
            "also updates alpha. "
            "Green compares interval methods for the same point model."
        ),
    )


def build_appendix_sp500():
    rows = read_csv(RESULTS_DIR / "sp500_point_model_selection_2023.csv")
    selected = read_json(CONFIG_DIR / "sp500_probabilistic_selection.json")
    selected_names = {
        selected["selected_classical_ml"]["Original"]["label"],
        selected["selected_dl"]["model_name"],
        "ZeroReturn",
    }
    parts = []
    for family_label, predicate, suffix in (
        (
            "classical and machine-learning",
            lambda row: row["Model"] not in DL_MODELS,
            "classical-ml",
        ),
        (
            "deep-learning",
            lambda row: row["Model"] in DL_MODELS,
            "dl",
        ),
    ):
        candidates = [row for row in rows if predicate(row)]
        candidates.sort(key=lambda row: number(row, "rmse"))
        maes = [number(row, "mae") for row in candidates]
        rmses = [number(row, "rmse") for row in candidates]
        times = [
            number(row, "selection_tuning_and_fit_seconds") for row in candidates
        ]
        body = []
        for rank, row in enumerate(candidates, 1):
            mae = number(row, "mae")
            rmse = number(row, "rmse")
            elapsed = number(row, "selection_tuning_and_fit_seconds")
            body.append(
                [
                    str(rank),
                    tex(model_label(row["Model"])),
                    highlight(fmt(mae, 6), green=is_min(mae, maes)),
                    highlight(
                        fmt(rmse, 6),
                        green=is_min(rmse, rmses),
                        bold=is_min(rmse, rmses),
                    ),
                    fmt(elapsed, 2) if elapsed is not None else "--",
                    r"\checkmark" if row["Model"] in selected_names else "",
                ]
            )
        parts.append(
            table(
                (
                    f"S\\&P 500 {family_label} point-model selection on the "
                    "2023 validation year."
                ),
                f"tab:appendix-sp500-selection-{suffix}",
                ["Rank", "Model", "MAE", "RMSE", "Selection time (s)", "Retained"],
                "rlrrrc",
                body,
                (
                    "Ranking uses RMSE. Selection time includes tuning plus "
                    "the final rolling validation stage for tree models, one "
                    "rolling validation stage for classical models, and the "
                    "saved automatic-search time for neural models. These "
                    "hardware-dependent timings are descriptive rather than "
                    "strictly comparable across families. "
                    "LightGBM and AutoLSTM are the family winners; the "
                    "zero-return forecast is retained as a benchmark."
                ),
                size=r"\scriptsize",
            )
        )
    return parts


def build_predictable_main():
    selection = read_csv(
        RESULTS_DIR / "predictable_conformal_point_selection.csv"
    )
    intervals = read_csv(RESULTS_DIR / "predictable_conformal_metrics.csv")
    dataset_order = ["US unemployment", "Conformal AR(1)", "Conformal sinusoid"]
    selected_models = {}
    for dataset in dataset_order:
        candidates = [row for row in selection if row["Dataset"] == dataset]
        selected_models[dataset] = min(
            candidates, key=lambda row: number(row, "rmse")
        )["Model"]
    method_order = [
        "Split conformal",
        "Online fixed-alpha",
        "Adaptive conformal inference",
    ]
    body = []
    for dataset in dataset_order:
        model = selected_models[dataset]
        candidates = [
            row
            for row in intervals
            if row["Dataset"] == dataset and row["Model"] == model
        ]
        available_methods = {row["Method"] for row in candidates}
        if not {"Split conformal", "Adaptive conformal inference"}.issubset(
            available_methods
        ):
            raise ValueError(
                f"Missing required interval rows for {dataset}/{model}: "
                f"{sorted(available_methods)}"
            )
        candidates.sort(key=lambda row: method_order.index(row["Method"]))
        errors = [number(row, "Coverage Error") for row in candidates]
        widths = [number(row, "Mean Interval Width") for row in candidates]
        scores = [number(row, "Mean Interval Score") for row in candidates]
        scientific = dataset == "Conformal sinusoid"
        label = {
            "US unemployment": "US unemployment",
            "Conformal AR(1)": "Long AR(1)",
            "Conformal sinusoid": "Deterministic sinusoid",
        }[dataset]
        for index, row in enumerate(candidates):
            error = number(row, "Coverage Error")
            width = number(row, "Mean Interval Width")
            score = number(row, "Mean Interval Score")
            body.append(
                [
                    tex(label) if index == 0 else "",
                    tex(model_label(model)) if index == 0 else "",
                    tex(interval_method_label(row["Method"])),
                    fmt(number(row, "Empirical Coverage"), 3),
                    highlight(fmt(error, 3), green=is_min(error, errors)),
                    highlight(
                        fmt(width, 3, scientific=scientific),
                        green=is_min(width, widths),
                    ),
                    highlight(
                        fmt(score, 3, scientific=scientific),
                        green=is_min(score, scores),
                        bold=is_min(score, scores),
                    ),
                ]
            )
    return table(
        (
            "Prediction intervals for the validation-selected point model in "
            "each predictable-series test."
        ),
        "tab:predictable-test-intervals",
        [
            "Series",
            "Point model",
            "Interval method",
            "Coverage",
            "Coverage error",
            "Mean width",
            "Interval score",
        ],
        "lllrrrr",
        body,
        (
            "Nominal coverage is 0.90. Selection uses point-forecast RMSE on "
            "the separate selection block. Calibration and test observations "
            "are not used for point-model selection. Lower coverage error and "
            "interval score are preferable; width is meaningful only jointly "
            "with coverage. The fixed residual-quantile benchmark reuses "
            "calibration residuals after refitting the point model and is "
            "descriptive rather than an ordinary split-conformal guarantee. "
            "Online fixed-alpha appends test residuals sequentially while "
            "holding alpha at 0.10; ACI uses the same expanding history and "
            "also updates alpha. "
            "Green compares interval methods for the same model."
        ),
    )


def build_appendix_predictable():
    rows = read_csv(RESULTS_DIR / "predictable_conformal_metrics.csv")
    dataset_order = ["US unemployment", "Conformal AR(1)", "Conformal sinusoid"]
    parts = []
    for dataset in dataset_order:
        candidates = [row for row in rows if row["Dataset"] == dataset]
        candidates.sort(
            key=lambda row: (
                number(row, "RMSE"),
                {
                    "Split conformal": 0,
                    "Online fixed-alpha": 1,
                    "Adaptive conformal inference": 2,
                }.get(row["Method"], 99),
            )
        )
        body = []
        for row in candidates:
            scientific = dataset == "Conformal sinusoid"
            body.append(
                [
                    tex(model_label(row["Model"])),
                    tex(interval_method_label(row["Method"])),
                    fmt(number(row, "Empirical Coverage"), 3),
                    fmt(number(row, "Coverage Error"), 3),
                    fmt(number(row, "Mean Interval Width"), 3, scientific),
                    fmt(number(row, "Mean Interval Score"), 3, scientific),
                    fmt(number(row, "RMSE"), 3, scientific),
                ]
            )
        parts.append(
            table(
                (
                    f"Complete 90\\% test-interval results for "
                    f"{tex(dataset)}."
                ),
                (
                    "tab:appendix-predictable-"
                    + dataset.lower().replace(" ", "-").replace("(", "").replace(")", "")
                ),
                [
                    "Point model",
                    "Interval method",
                    "Coverage",
                    "Coverage error",
                    "Mean width",
                    "Interval score",
                    "RMSE",
                ],
                "llrrrrr",
                body,
                (
                    "Coverage, width, interval score, and point RMSE are "
                    "computed on the untouched test block. No test ranking was "
                    "used for model selection. The fixed residual-quantile "
                    "rows are descriptive because the point model is refitted "
                    "between calibration and test."
                ),
                size=r"\scriptsize",
            )
        )
    return parts


def exogenous_carried_models(config, validation_rows):
    output = {}
    for profile in PROFILE_ORDER:
        if profile == "full_public_native":
            models = [
                row["Model"]
                for row in validation_rows
                if row["Profile"] == profile
            ]
        else:
            models = [
                config["best_tree"][profile]["model"],
                "AutoARIMAX",
                config["best_dl"][profile]["model"],
            ]
        output[profile] = models
    return output


def build_exogenous_main():
    validation = read_csv(
        METRICS_DIR / "exogenous_models_metrics_validation.csv"
    )
    test = read_csv(METRICS_DIR / "exogenous_models_metrics_test.csv")
    config = read_json(CONFIG_DIR / "exogenous_validation_configs.json")
    carried = exogenous_carried_models(config, validation)
    validation_body = []
    test_body = []
    for profile in PROFILE_ORDER:
        val_rows = [
            one(validation, Profile=profile, Model=model, Split="validation")
            for model in carried[profile]
        ]
        val_maes = [number(row, "mae") for row in val_rows]
        val_rmses = [number(row, "rmse") for row in val_rows]
        for index, row in enumerate(val_rows):
            validation_body.append(
                [
                    tex(PROFILE_LABELS[profile]) if index == 0 else "",
                    tex(row["Model Family"]),
                    tex(model_label(row["Model"])),
                    highlight(
                        fmt(number(row, "mae")),
                        green=is_min(number(row, "mae"), val_maes),
                        bold=is_min(number(row, "mae"), val_maes),
                    ),
                    fmt(number(row, "absolute_error_sd")),
                    highlight(
                        fmt(number(row, "rmse")),
                        green=is_min(number(row, "rmse"), val_rmses),
                    ),
                    (
                        r"\checkmark"
                        if row["Model"] == config["winners"][profile]["model"]
                        else ""
                    ),
                ]
            )

        test_rows = [
            one(test, Profile=profile, Model=model, Split="test")
            for model in carried[profile]
        ]
        test_maes = [number(row, "mae") for row in test_rows]
        test_sds = [number(row, "absolute_error_sd") for row in test_rows]
        test_rmses = [number(row, "rmse") for row in test_rows]
        test_biases = [
            abs(number(row, "forecast_bias")) for row in test_rows
        ]
        for index, row in enumerate(test_rows):
            test_body.append(
                [
                    tex(PROFILE_LABELS[profile]) if index == 0 else "",
                    tex(row["Model Family"]),
                    tex(model_label(row["Model"])),
                    highlight(
                        fmt(number(row, "mae")),
                        green=is_min(number(row, "mae"), test_maes),
                        bold=is_min(number(row, "mae"), test_maes),
                    ),
                    highlight(
                        fmt(number(row, "absolute_error_sd")),
                        green=is_min(
                            number(row, "absolute_error_sd"), test_sds
                        ),
                    ),
                    highlight(
                        fmt(number(row, "rmse")),
                        green=is_min(number(row, "rmse"), test_rmses),
                    ),
                    highlight(
                        fmt(number(row, "forecast_bias"), 1),
                        green=is_min(
                            abs(number(row, "forecast_bias")),
                            test_biases,
                        ),
                    ),
                ]
            )
    validation_table = table(
        (
            "Compact validation selection for the Argentine inflation "
            "forecasting exercise."
        ),
        "tab:exogenous-validation-selection",
        ["Profile", "Family", "Model", "MAE", "SD($|e|$)", "RMSE", "Overall choice"],
        "lllrrrc",
        validation_body,
        (
            "Errors and the standard deviation of absolute errors are in "
            "monthly inflation percentage points over 12 validation months. "
            "The selection criterion is MAE. For complete profiles, the best "
            "tree, AutoARIMAX, and best deep-learning model are shown; both "
            "native-missing tree candidates are shown for the public panel. "
            "Trees and AutoARIMAX update their parameters monthly. Each neural "
            "candidate is fitted once before the block and produces 12 rolling "
            "one-month-ahead forecasts with fixed weights and updated observed "
            "history."
        ),
    )
    test_table = table(
        (
            "Final Argentine monthly-inflation test errors for the "
            "validation-retained specifications."
        ),
        "tab:exogenous-test-selected",
        ["Profile", "Family", "Model", "MAE", "SD($|e|$)", "RMSE", "Bias (\\%)"],
        "lllrrrr",
        test_body,
        (
            "MAE, RMSE, and the standard deviation of absolute errors are in "
            "monthly inflation percentage points over the 12 test months "
            "(August 2023 to July 2024); bias is a percentage. Green gives "
            "descriptive within-profile "
            "test minima, while bold identifies the within-profile MAE "
            "minimum, the experiment's primary metric. This formatting does "
            "not change validation selection. Runtime is "
            "omitted because the saved definitions are not comparable across "
            "families."
        ),
    )
    return [validation_table, test_table]


def build_exogenous_shap():
    rows = read_csv(METRICS_DIR / "exogenous_tree_shap_variable_summary.csv")
    body = []
    for profile in PROFILE_ORDER:
        candidates = sorted(
            [row for row in rows if row["Profile"] == profile],
            key=lambda row: number(row, "Importance Share"),
            reverse=True,
        )[:3]
        for rank, row in enumerate(candidates, 1):
            body.append(
                [
                    tex(PROFILE_LABELS[profile]) if rank == 1 else "",
                    str(rank),
                    tex(row["Variable Group"]),
                    fmt(100.0 * number(row, "Importance Share"), 1) + r"\%",
                    str(int(number(row, "Origins"))),
                ]
            )
    return table(
        (
            "Largest validation-period TreeSHAP importance shares by "
            "inflation-data profile."
        ),
        "tab:exogenous-shap-top",
        ["Profile", "Rank", "Variable group", "Importance share", "Origins"],
        "lrlrr",
        body,
        (
            "Shares normalize mean absolute SHAP values across variables and "
            "average the 12 rolling validation origins for each selected tree. "
            "They describe predictive attribution on the transformed target "
            "scale and should not be read as causal effects."
        ),
    )


def build_appendix_exogenous():
    validation = read_csv(
        METRICS_DIR / "exogenous_models_metrics_validation.csv"
    )
    config = read_json(CONFIG_DIR / "exogenous_validation_configs.json")
    shap_rows = read_csv(
        METRICS_DIR / "exogenous_tree_shap_variable_summary.csv"
    )
    parts = []
    for profile in PROFILE_ORDER:
        neural_screen = config.get("dl_timing", {}).get(profile, [])
        if neural_screen:
            neural_screen = sorted(
                neural_screen,
                key=lambda row: number(row, "Best Validation Loss"),
            )
            neural_maes = [
                number(row, "Best Validation Loss") for row in neural_screen
            ]
            neural_times = [
                number(row, "Time Elapsed") for row in neural_screen
            ]
            retained_neural = config["best_dl"][profile]["model"]
            neural_body = []
            for rank, row in enumerate(neural_screen, 1):
                mae = number(row, "Best Validation Loss")
                elapsed = number(row, "Time Elapsed")
                neural_body.append(
                    [
                        str(rank),
                        tex(model_label(row["Model"])),
                        str(int(number(row, "Completed Configurations"))),
                        highlight(
                            fmt(mae),
                            green=is_min(mae, neural_maes),
                            bold=is_min(mae, neural_maes),
                        ),
                        highlight(
                            fmt(elapsed, 2),
                            green=is_min(elapsed, neural_times),
                            bold=is_min(elapsed, neural_times),
                        ),
                        r"\checkmark" if row["Model"] == retained_neural else "",
                    ]
                )
            parts.append(
                table(
                    (
                        "Exogenous neural-architecture validation screen for "
                        f"{tex(PROFILE_LABELS[profile])}."
                    ),
                    (
                        "tab:appendix-exogenous-neural-screen-"
                        f"{profile.replace('_', '-')}"
                    ),
                    [
                        "Rank",
                        "Architecture",
                        "Completed configurations",
                        "Best MAE",
                        "Search time (s)",
                        "Retained",
                    ],
                    "rlrrrc",
                    neural_body,
                    (
                        "Each architecture receives eight seed-67 TPE "
                        "configurations, including three startup "
                        "configurations. Best MAE is the lowest mean absolute "
                        "error over the same 12 rolling one-step validation "
                        "origins. Search time is architecture-specific and "
                        "includes the eight fitted configurations. The "
                        "retained profile--architecture combination was "
                        "fixed before test."
                    ),
                    size=r"\scriptsize",
                )
            )

        candidates = sorted(
            [row for row in validation if row["Profile"] == profile],
            key=lambda row: number(row, "mae"),
        )
        maes = [number(row, "mae") for row in candidates]
        rmses = [number(row, "rmse") for row in candidates]
        times = [number(row, "Time Elapsed") for row in candidates]
        body = []
        for rank, row in enumerate(candidates, 1):
            mae = number(row, "mae")
            rmse = number(row, "rmse")
            elapsed = number(row, "Time Elapsed")
            body.append(
                [
                    str(rank),
                    tex(row["Model Family"]),
                    tex(model_label(row["Model"])),
                    tex(row["Target Transformation"]),
                    highlight(
                        fmt(mae),
                        green=is_min(mae, maes),
                        bold=is_min(mae, maes),
                    ),
                    fmt(number(row, "absolute_error_sd")),
                    highlight(fmt(rmse), green=is_min(rmse, rmses)),
                    highlight(
                        fmt(elapsed, 2),
                        green=is_min(elapsed, times),
                        bold=is_min(elapsed, times),
                    ),
                    (
                        r"\checkmark"
                        if row["Model"] == config["winners"][profile]["model"]
                        else ""
                    ),
                ]
            )
        parts.append(
            table(
                (
                    f"Family-level exogenous-model validation results for "
                    f"{tex(PROFILE_LABELS[profile])}."
                ),
                f"tab:appendix-exogenous-validation-{profile.replace('_', '-')}",
                [
                    "Rank",
                    "Family",
                    "Model",
                    "Target",
                    "MAE",
                    "SD($|e|$)",
                    "RMSE",
                    "Time (s)",
                    "Overall MAE leader",
                ],
                "rlllrrrrc",
                body,
                (
                    "Rank and the overall-leader indicator use MAE over 12 "
                    "validation months. This indicator is not the complete "
                    "retention rule: the best tree, AutoARIMAX when admissible, "
                    "and two neural shortlist members advance under separate "
                    "family-specific rules. Trees and AutoARIMAX update their "
                    "parameters monthly. Neural models are fitted once before "
                    "the block and then issue 12 rolling one-month-ahead "
                    "forecasts with fixed weights and updated observed history. "
                    "Errors are monthly inflation percentage points. Timing "
                    "definitions differ by family and are reported descriptively."
                ),
                size=r"\scriptsize",
            )
        )

        candidates = sorted(
            [row for row in shap_rows if row["Profile"] == profile],
            key=lambda row: number(row, "Importance Share"),
            reverse=True,
        )
        body = []
        for rank, row in enumerate(candidates, 1):
            body.append(
                [
                    str(rank),
                    tex(row["Variable Group"]),
                    fmt(number(row, "Mean_Absolute_SHAP"), 3),
                    fmt(number(row, "Mean_SHAP"), 3),
                    fmt(number(row, "Positive_Share"), 3),
                    fmt(100.0 * number(row, "Importance Share"), 1) + r"\%",
                ]
            )
        parts.append(
            table(
                (
                    f"Complete validation TreeSHAP summary for "
                    f"{tex(PROFILE_LABELS[profile])}."
                ),
                f"tab:appendix-exogenous-shap-{profile.replace('_', '-')}",
                [
                    "Rank",
                    "Variable group",
                    "Mean $|\\phi|$",
                    "Mean $\\phi$",
                    "Positive share",
                    "Importance share",
                ],
                "rlrrrr",
                body,
                (
                    "Values average 12 rolling validation origins for the "
                    "selected tree. Attributions use the transformed target "
                    "scale and are predictive rather than causal."
                ),
                size=r"\scriptsize",
            )
        )
    return parts


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write(
        "main_manual_dl_selection.tex",
        [build_manual_dl_top_two()],
    )
    write(
        "main_manual_monthly_test.tex",
        [
            build_manual_test_table(
                MONTHLY_DATASETS,
                "monthly",
                "tab:manual-monthly-test",
            )
        ],
    )
    write(
        "main_manual_quarterly_test.tex",
        [
            build_manual_test_table(
                QUARTERLY_DATASETS,
                "quarterly",
                "tab:manual-quarterly-test",
            )
        ],
    )
    write("main_tree_transformation.tex", [build_tree_transformation()])
    write("main_global_selection_test.tex", build_global_main())
    write("main_sp500_intervals.tex", [build_sp500_main()])
    write("main_predictable_intervals.tex", [build_predictable_main()])
    write("main_exogenous_selection_test.tex", build_exogenous_main())
    write("main_exogenous_shap.tex", [build_exogenous_shap()])
    write("appendix_manual_dl_all.tex", build_appendix_manual())
    write("appendix_global_all.tex", build_appendix_global())
    write("appendix_sp500_selection.tex", build_appendix_sp500())
    write("appendix_predictable_all.tex", build_appendix_predictable())
    write("appendix_exogenous_all.tex", build_appendix_exogenous())


if __name__ == "__main__":
    main()
