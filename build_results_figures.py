"""Build lightweight Results figures from persisted forecast CSVs.

This script does not fit or tune any model. It reads the saved global test
forecasts and draws the two specifications retained after validation for the
first series in each synthetic panel.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


THESIS_DIR = Path(__file__).resolve().parent
METRICS_DIR = THESIS_DIR / "metrics"
OUTPUT_DIR = THESIS_DIR / "imgs" / "global_models"
PREDICTABLE_OUTPUT_DIR = (
    THESIS_DIR / "imgs" / "predictable_series_conformal"
)

PANELS = {
    "autoregressive": {
        "title": "AR(1)",
        "forest": "Global_XGBRandomForest",
        "forest_label": "Global XGBRandomForest",
        "neural": "AutoRNN",
        "output": "global_synthetic_ar1_selected_test_first_series.jpg",
    },
    "pseudo_periodic": {
        "title": "Pseudo-periodic",
        "forest": "Global_XGBRandomForest",
        "forest_label": "Global XGBRandomForest",
        "neural": "AutoGRU",
        "output": (
            "global_synthetic_pseudo_periodic_selected_test_first_series.jpg"
        ),
    },
    "sinusoidal_trend_noise": {
        "title": "Sinusoid, trend, and noise",
        "forest": "Global_XGBRandomForest_AutoStationary",
        "forest_label": "Global XGBRandomForest (AutoStationary)",
        "neural": "AutoPatchTST",
        "output": (
            "global_synthetic_sinusoidal_trend_noise_selected_test_"
            "first_series.jpg"
        ),
    },
}


def read_first_series(panel):
    path = METRICS_DIR / f"global_models_predictions_test_{panel}.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    first_id = rows[0]["unique_id"]
    return [row for row in rows if row["unique_id"] == first_id]


def build_panel(panel, specification):
    rows = read_first_series(panel)
    steps = list(range(1, len(rows) + 1))
    actual = [float(row["y"]) for row in rows]
    forest = [float(row[specification["forest"]]) for row in rows]
    neural = [float(row[specification["neural"]]) for row in rows]

    figure, axis = plt.subplots(figsize=(9.4, 4.8))
    axis.plot(
        steps,
        actual,
        color="#202020",
        linewidth=2.3,
        marker="o",
        markersize=4.5,
        label="Observed value",
        zorder=3,
    )
    axis.plot(
        steps,
        forest,
        color="#2878B5",
        linewidth=2.0,
        marker="s",
        markersize=4,
        label=specification["forest_label"],
    )
    axis.plot(
        steps,
        neural,
        color="#C44E52",
        linewidth=2.0,
        marker="^",
        markersize=4,
        label=specification["neural"],
    )
    axis.set(
        title=f"First series: {specification['title']} panel",
        xlabel="Forecast step",
        ylabel="Simulated value",
        xticks=steps,
    )
    axis.grid(alpha=0.20, linewidth=0.7)
    axis.legend(frameon=False, fontsize=9, ncol=1)
    figure.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / specification["output"]
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def rolling_mean(values, window):
    output = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        output.append(running / min(index + 1, window))
    return output


def build_predictable_coverage():
    path = THESIS_DIR / "results" / "predictable_conformal_intervals.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    panels = [
        ("US unemployment", "AutoARIMA", 12, "US unemployment"),
        ("Conformal AR(1)", "AR1OLS", 50, "Long AR(1)"),
        (
            "Conformal sinusoid",
            "AutoTiDE",
            50,
            "Deterministic sinusoid",
        ),
    ]
    methods = [
        ("Split conformal", "#2878B5", "-"),
        ("Adaptive conformal inference", "#C44E52", "--"),
    ]
    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9.4, 8.2),
        sharex=False,
        sharey=True,
    )
    for axis, (dataset, model, window, title) in zip(axes, panels):
        for method, color, linestyle in methods:
            selected = [
                row
                for row in rows
                if row["Dataset"] == dataset
                and row["Model"] == model
                and row["Method"] == method
            ]
            hits = [
                float(row["lower"])
                <= float(row["y"])
                <= float(row["upper"])
                for row in selected
            ]
            coverage = rolling_mean(
                [float(hit) for hit in hits], window
            )[window - 1 :]
            axis.plot(
                range(window, len(hits) + 1),
                coverage,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=method,
            )
        axis.axhline(
            0.90,
            color="#303030",
            linestyle=":",
            linewidth=1.2,
            label="90% nominal coverage",
        )
        axis.set_title(
            f"{title}: rolling {window}-origin coverage",
            fontsize=11,
        )
        axis.set_ylabel("Coverage")
        axis.set_ylim(0.0, 1.04)
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    axes[-1].set_xlabel("Test origin")
    figure.tight_layout()

    PREDICTABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = (
        PREDICTABLE_OUTPUT_DIR
        / "principal_models_rolling_coverage.jpg"
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


if __name__ == "__main__":
    for panel_name, panel_specification in PANELS.items():
        print(build_panel(panel_name, panel_specification))
    print(build_predictable_coverage())
