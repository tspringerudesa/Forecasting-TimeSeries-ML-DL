"""Shared publication metadata, plotting labels, and artifact exporters.

The forecasting code uses compact machine identifiers internally.  This module
is the single boundary where those identifiers are converted into the titles,
units, captions, and filesystem-safe names used by notebooks and the paper.
"""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd


THESIS_DIR = Path(__file__).resolve().parent
IMAGES_DIR = THESIS_DIR / "imgs"
RESULTS_DIR = THESIS_DIR / "results"


DATASET_METADATA = {
    "emae": {
        "display_name": "Argentina EMAE",
        "full_name": "Argentina Monthly Estimator of Economic Activity (EMAE)",
        "frequency": "Monthly",
        "units": "Index (2004=100)",
        "seasonal_adjustment": "Original series (not seasonally adjusted)",
        "y_axis_label": "EMAE index (2004=100)",
        "series_name": "Argentina EMAE",
        "source": "INDEC / Datos Argentina",
        "source_id": "143.3_NO_PR_2004_A_21",
        "slug": "argentina_emae",
    },
    "ecai_us": {
        "display_name": "US Coincident Economic Activity Index",
        "full_name": "Coincident Economic Activity Index for the United States",
        "frequency": "Monthly",
        "units": "Index (2007=100)",
        "seasonal_adjustment": "Seasonally adjusted",
        "y_axis_label": "Coincident activity index (2007=100)",
        "series_name": "US Coincident Economic Activity Index",
        "source": "Federal Reserve Bank of Philadelphia via FRED",
        "source_id": "USPHCI",
        "slug": "us_coincident_activity_index",
    },
    "unemployment_us": {
        "display_name": "US Unemployment Rate",
        "full_name": "Unemployment Rate for the United States",
        "frequency": "Monthly",
        "units": "Percent",
        "seasonal_adjustment": "Seasonally adjusted",
        "y_axis_label": "Unemployment rate (%)",
        "series_name": "US Unemployment Rate",
        "source": "US Bureau of Labor Statistics via FRED",
        "source_id": "UNRATE",
        "slug": "us_unemployment_rate",
    },
    "real_gdp_argentina": {
        "display_name": "Argentina Real GDP",
        "full_name": "Real Gross Domestic Product for Argentina",
        "frequency": "Quarterly",
        "units": "Millions of Domestic Currency",
        "seasonal_adjustment": "Not seasonally adjusted",
        "y_axis_label": "Millions of domestic currency",
        "series_name": "Argentina Real GDP",
        "source": "International Monetary Fund via FRED",
        "source_id": "NGDPRNSAXDCARQ",
        "slug": "argentina_real_gdp",
    },
    "real_gdp_us": {
        "display_name": "US Real GDP",
        "full_name": "Real Gross Domestic Product for the United States",
        "frequency": "Quarterly",
        "units": "Billions of Chained 2017 Dollars",
        "seasonal_adjustment": "Not seasonally adjusted",
        "y_axis_label": "Billions of chained 2017 dollars",
        "series_name": "US Real GDP",
        "source": "US Bureau of Economic Analysis via FRED",
        "source_id": "ND000334Q",
        "slug": "us_real_gdp",
    },
    "sp500_returns": {
        "display_name": "S&P 500 Daily Log Returns",
        "full_name": "S&P 500 Daily Log Returns",
        "frequency": "Observed trading day",
        "units": "Log return",
        "seasonal_adjustment": "Not applicable",
        "y_axis_label": "Daily log return",
        "series_name": "S&P 500",
        "source": "Yahoo Finance",
        "source_id": "^GSPC",
        "slug": "sp500_daily_log_returns",
    },
    "autoregressive": {
        "display_name": "Global Synthetic AR(1)",
        "full_name": "Global Synthetic Autoregressive Process (AR(1), phi=0.75)",
        "frequency": "Generated regular step",
        "units": "Simulated value",
        "seasonal_adjustment": "Not applicable",
        "y_axis_label": "Simulated AR(1) value",
        "series_name": "Synthetic AR(1)",
        "source": "Thesis simulation",
        "source_id": "AR(1)",
        "slug": "global_synthetic_ar1",
    },
    "pseudo_periodic": {
        "display_name": "Global Synthetic Pseudo-Periodic Process",
        "full_name": "Global Synthetic Pseudo-Periodic Process",
        "frequency": "Generated regular step",
        "units": "Simulated value",
        "seasonal_adjustment": "Not applicable",
        "y_axis_label": "Simulated pseudo-periodic value",
        "series_name": "Synthetic pseudo-periodic process",
        "source": "Thesis simulation",
        "source_id": "PseudoPeriodic",
        "slug": "global_synthetic_pseudo_periodic",
    },
    "sinusoidal_trend_noise": {
        "display_name": "Global Synthetic Sinusoid with Trend and Noise",
        "full_name": "Global Synthetic Sinusoidal Process with Trend and Gaussian Noise",
        "frequency": "Generated regular step",
        "units": "Simulated value",
        "seasonal_adjustment": "Not applicable",
        "y_axis_label": "Simulated sinusoid-plus-trend value",
        "series_name": "Synthetic sinusoid with trend and noise",
        "source": "Thesis simulation",
        "source_id": "Sinusoidal+trend+noise",
        "slug": "global_synthetic_sinusoidal_trend_noise",
    },
    "argentina_inflation": {
        "display_name": "Argentina Monthly CPI Inflation",
        "full_name": "Argentina Consumer Price Inflation",
        "frequency": "Monthly",
        "units": "Monthly percent change",
        "seasonal_adjustment": "Not seasonally adjusted",
        "y_axis_label": "Monthly CPI inflation (%)",
        "series_name": "Argentina CPI inflation",
        "source": "Forte public-data reconstruction",
        "source_id": "CPI growth",
        "slug": "argentina_monthly_cpi_inflation",
    },
    "ar1_conformal": {
        "display_name": "Conformal Synthetic AR(1)",
        "full_name": "Conformal Synthetic Autoregressive Process (AR(1), phi=0.50)",
        "frequency": "Generated regular step",
        "units": "Simulated value",
        "seasonal_adjustment": "Not applicable",
        "y_axis_label": "Simulated AR(1) value",
        "series_name": "Conformal synthetic AR(1)",
        "source": "Thesis simulation",
        "source_id": "AR(1), phi=0.50, innovation SD=1",
        "slug": "conformal_synthetic_ar1",
    },
    "sinusoidal_conformal": {
        "display_name": "Conformal Synthetic Sinusoid",
        "full_name": (
            "Deterministic Synthetic Sinusoid "
            "(amplitude=1, TimeSynth frequency=0.25, about 20 observations per cycle)"
        ),
        "frequency": "Generated regular step",
        "units": "Simulated value",
        "seasonal_adjustment": "Not applicable",
        "y_axis_label": "Deterministic sinusoidal value",
        "series_name": "Conformal synthetic sinusoid",
        "source": "Thesis simulation",
        "source_id": (
            "Sinusoid, amplitude=1, TimeSynth frequency=0.25, "
            "five observations per TimeSynth time unit"
        ),
        "slug": "conformal_synthetic_sinusoid",
    },
}


DATASET_ALIASES = {
    "US unemployment": "unemployment_us",
    "AR(1)": "ar1_conformal",
    "Conformal AR(1)": "ar1_conformal",
    "Conformal sinusoid": "sinusoidal_conformal",
}


def metadata_for(dataset_name: str) -> Mapping[str, str]:
    """Return publication metadata for a dataset key or supported alias."""
    canonical = DATASET_ALIASES.get(str(dataset_name), str(dataset_name))
    if canonical not in DATASET_METADATA:
        raise KeyError(
            f"No reporting metadata for {dataset_name!r}; "
            f"available keys are {sorted(DATASET_METADATA)}"
        )
    return DATASET_METADATA[canonical]


def dataset_descriptor(dataset_name: str) -> str:
    """Return the complete title/frequency/units/adjustment description."""
    metadata = metadata_for(dataset_name)
    return (
        f"{metadata['display_name']}; {metadata['frequency']}; "
        f"{metadata['units']}; {metadata['seasonal_adjustment']}"
    )


def display_series_name(dataset_name: str, series_name: Optional[str] = None) -> str:
    """Replace internal underscore identifiers with readable series labels."""
    metadata = metadata_for(dataset_name)
    if series_name is None:
        return metadata["series_name"]
    raw = str(series_name)
    global_match = re.search(r"_(\d{3})$", raw)
    if global_match and dataset_name in {
        "autoregressive",
        "pseudo_periodic",
        "sinusoidal_trend_noise",
    }:
        return f"{metadata['series_name']} {int(global_match.group(1)) + 1}"
    return metadata["series_name"]


def slugify(value: str) -> str:
    """Return a stable lowercase filename component."""
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return value or "artifact"


def figure_path(
    workflow: str,
    dataset_name: str,
    *parts: str,
    extension: str = "jpg",
) -> Path:
    """Build a consistent path under ``thesis/imgs/<workflow>``."""
    metadata = metadata_for(dataset_name)
    output_dir = IMAGES_DIR / slugify(workflow)
    stem_parts = [
        metadata["slug"],
        *(slugify(part) for part in parts if part),
    ]
    return output_dir / ("_".join(stem_parts) + f".{extension.lstrip('.')}")


def save_plotly_figure(
    figure,
    path: Path,
    enabled: bool,
    *,
    width: int = 1200,
    height: int = 650,
    scale: float = 2.0,
) -> Optional[Path]:
    """Optionally export a Plotly figure using the installed Kaleido backend."""
    if not enabled:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_image(
        str(path),
        width=int(width),
        height=int(height),
        scale=float(scale),
    )
    return path


def save_matplotlib_figure(
    figure,
    path: Path,
    enabled: bool,
    *,
    dpi: int = 300,
) -> Optional[Path]:
    """Optionally export a Matplotlib figure with tight bounds."""
    if not enabled:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=int(dpi))
    return path


def apply_dataset_labels(
    figure,
    dataset_name: str,
    *,
    title_suffix: Optional[str] = None,
    x_axis_title: Optional[str] = None,
):
    """Apply consistent publication labels and margins to a Plotly figure."""
    metadata = metadata_for(dataset_name)
    title = metadata["display_name"]
    if title_suffix:
        title = f"{title}<br><sup>{title_suffix}</sup>"
    figure.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        xaxis_title=x_axis_title or (
            "Generated time step"
            if metadata["frequency"] == "Generated regular step"
            else "Date"
        ),
        yaxis_title=metadata["y_axis_label"],
        margin={"l": 90, "r": 35, "t": 90, "b": 65},
        template="plotly_white",
    )
    return figure


def apply_matplotlib_dataset_labels(
    figure,
    dataset_name: str,
    *,
    title_suffix: Optional[str] = None,
    x_axis_title: Optional[str] = None,
):
    """Apply publication labels to a Matplotlib forecast figure."""
    metadata = metadata_for(dataset_name)
    title = metadata["display_name"]
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    axis = figure.axes[0]
    axis.set_title(title, loc="left", fontsize=12)
    axis.set_xlabel(
        x_axis_title
        or (
            "Generated time step"
            if metadata["frequency"] == "Generated regular step"
            else "Date"
        )
    )
    axis.set_ylabel(metadata["y_axis_label"])
    figure.tight_layout()
    return figure


def humanize_results_table(
    result_table: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Use readable labels and omit a redundant single-series column."""
    table = result_table.copy()
    if "Series Name" in table:
        table["Series Name"] = [
            display_series_name(dataset_name, value)
            for value in table["Series Name"]
        ]
        if table["Series Name"].nunique(dropna=False) == 1:
            table = table.drop(columns="Series Name")
    return table


def _latex_label_component(value: str) -> str:
    return slugify(value).replace("_", "-")


_LOWER_IS_BETTER_PATTERNS = (
    r"\bmae\b",
    r"\bmse\b",
    r"\brmse\b",
    r"\bmape\b",
    r"\bsmape\b",
    r"\bmase\b",
    r"\bloss\b",
    r"forecast[\s_-]*bias",
    r"coverage[\s_-]*error",
    r"interval[\s_-]*score",
    r"absolute[\s_-]*error",
    r"\brv\b",
    r"\brse\b",
)


def _numeric_table_value(value) -> float:
    """Parse a numeric or percentage-formatted table cell for comparison."""
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return float("nan")
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("%", "")
        if cleaned in {"", "--", "NaN", "nan", "None"}:
            return float("nan")
        try:
            return float(cleaned)
        except ValueError:
            # Composite summary cells such as ``0.1234 (0.0456)`` compare
            # their leading estimate; the parenthesized value is dispersion.
            leading_number = re.match(
                r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
                cleaned,
            )
            return (
                float(leading_number.group(0))
                if leading_number
                else float("nan")
            )
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_lower_is_better_column(column) -> bool:
    """Identify accuracy, bias, forecastability, and runtime columns."""
    name = str(column).strip().lower()
    if any(re.search(pattern, name) for pattern in _LOWER_IS_BETTER_PATTERNS):
        return True
    return (
        "time" in name
        or "seconds" in name
        or name.endswith(" duration")
        or name.endswith(" runtime")
    )


def _is_bold_best_column(column) -> bool:
    """Bold the primary accuracy metric and total elapsed runtime."""
    name = re.sub(r"\s+", " ", str(column).strip().lower())
    return bool(re.search(r"\brmse\b", name)) or name in {
        "time elapsed",
        "total time elapsed",
        "total elapsed time",
        "end-to-end time",
        "end-to-end runtime",
    }


def _default_comparison_groups(table: pd.DataFrame) -> list[str]:
    """Infer blocks within which models are genuinely comparable."""
    groups = [
        column
        for column in (
            "Dataset",
            "DGP",
            "Profile",
            "Series Name",
            "Series",
            "Series and measurement",
            "Process",
        )
        if column in table.columns
    ]
    probabilistic_columns = {
        str(column).lower() for column in table.columns
    }
    if "Method" in table.columns and any(
        "coverage" in column or "interval" in column
        for column in probabilistic_columns
    ):
        groups.append("Method")
    return groups


def _best_metric_cells(
    table: pd.DataFrame,
    comparison_group_columns: Optional[Sequence[str]] = None,
) -> tuple[set[tuple[int, object]], set[tuple[int, object]]]:
    """Return green and bold cell coordinates for lower-is-better metrics."""
    metric_columns = [
        column for column in table.columns
        if _is_lower_is_better_column(column)
    ]
    if not metric_columns or table.empty:
        return set(), set()

    if comparison_group_columns is None:
        group_columns = _default_comparison_groups(table)
    else:
        group_columns = [
            column for column in comparison_group_columns
            if column in table.columns
        ]

    if group_columns:
        groups = table.groupby(
            group_columns, sort=False, dropna=False
        ).groups.values()
    else:
        groups = [table.index]

    green_cells: set[tuple[int, object]] = set()
    bold_cells: set[tuple[int, object]] = set()
    index_positions = {
        index_value: position
        for position, index_value in enumerate(table.index)
    }
    for group_index in groups:
        positions = [index_positions[index_value] for index_value in group_index]
        for column in metric_columns:
            values = [
                _numeric_table_value(table.iloc[position][column])
                for position in positions
            ]
            finite = [abs(value) for value in values if math.isfinite(value)]
            if not finite:
                continue
            best = min(finite)
            for position, value in zip(positions, values):
                if math.isfinite(value) and math.isclose(
                    abs(value), best, rel_tol=1e-12, abs_tol=1e-15
                ):
                    coordinate = (position, column)
                    green_cells.add(coordinate)
                    if _is_bold_best_column(column):
                        bold_cells.add(coordinate)
    return green_cells, bold_cells


def _format_latex_cell(value, float_format) -> str:
    """Format one value before pandas performs LaTeX escaping."""
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return "--"
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return float_format(value)
    return str(value)


def publication_latex(
    table: pd.DataFrame,
    *,
    caption: str,
    short_caption: Optional[str] = None,
    label: str,
    float_format=None,
    highlight_best: bool = True,
    comparison_group_columns: Optional[Sequence[str]] = None,
) -> str:
    """Render a compact width-safe table with consistent best-cell emphasis."""
    if float_format is None:
        float_format = (
            lambda value: f"{value:.6f}" if abs(value) < 100 else f"{value:.2f}"
        )
    source_table = table.reset_index(drop=True).copy()
    if highlight_best:
        green_cells, bold_cells = _best_metric_cells(
            source_table,
            comparison_group_columns=comparison_group_columns,
        )
    else:
        green_cells, bold_cells = set(), set()

    formatted_table = source_table.copy().astype(object)
    replacements = {}
    marker_number = 0
    for row_position in range(len(source_table)):
        for column in source_table.columns:
            rendered = _format_latex_cell(
                source_table.iloc[row_position][column], float_format
            )
            coordinate = (row_position, column)
            if coordinate in green_cells:
                command = rf"\textcolor{{metricgreen}}{{{rendered}}}"
                if coordinate in bold_cells:
                    command = rf"\textcolor{{metricgreen}}{{\textbf{{{rendered}}}}}"
                marker = f"LATEXHIGHLIGHTCELL{marker_number:06d}"
                marker_number += 1
                replacements[marker] = command
                rendered = marker
            formatted_table.iat[
                row_position, formatted_table.columns.get_loc(column)
            ] = rendered

    column_format = "".join(
        "r"
        if pd.api.types.is_numeric_dtype(source_table[column])
        or source_table[column].map(_numeric_table_value).notna().all()
        else "l"
        for column in source_table.columns
    )
    safe_caption = str(caption).replace("&", r"\&").replace("%", r"\%")
    latex_caption = safe_caption
    if short_caption is not None:
        safe_short_caption = (
            str(short_caption).replace("&", r"\&").replace("%", r"\%")
        )
        latex_caption = (safe_caption, safe_short_caption)
    latex = formatted_table.to_latex(
        index=False,
        na_rep="--",
        caption=latex_caption,
        label=label,
        position="htbp",
        escape=True,
        column_format=column_format,
    )
    for marker, command in replacements.items():
        latex = latex.replace(marker, command)
    latex = latex.replace(
        "\\centering\n",
        "\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n",
        1,
    )
    latex = latex.replace(
        "\\begin{tabular}",
        "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}",
        1,
    )
    latex = latex.replace(
        "\\end{tabular}",
        "\\end{tabular}\n\\end{adjustbox}",
        1,
    )
    return latex


def export_publication_table(
    table: pd.DataFrame,
    *,
    output_file: Path,
    dataset_name: str,
    method_name: str,
    split: str,
    target_transformation: Optional[str] = None,
    float_format=None,
) -> Path:
    """Write one self-describing, width-safe LaTeX table."""
    descriptor = dataset_descriptor(dataset_name)
    target = (
        f"; target representation: {target_transformation}"
        if target_transformation
        else ""
    )
    caption = (
        f"{descriptor}. {method_name} {split} performance{target}."
    )
    label_parts = [
        "tab",
        _latex_label_component(method_name),
        _latex_label_component(split),
        _latex_label_component(dataset_name),
    ]
    if target_transformation:
        label_parts.append(_latex_label_component(target_transformation))
    latex = publication_latex(
        table,
        caption=caption,
        label=":".join(label_parts),
        float_format=float_format,
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(latex, encoding="utf-8")
    return output_file


def export_publication_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    output_file: Path,
    dataset_name: str,
    method_name: str,
    split: str,
    float_format=None,
) -> Path:
    """Write one width-safe LaTeX table per target representation."""
    sections = [
        export_table_text(
            table,
            dataset_name=dataset_name,
            method_name=method_name,
            split=split,
            target_transformation=target,
            float_format=float_format,
        )
        for target, table in tables.items()
    ]
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n\n".join(sections), encoding="utf-8")
    return output_file


def export_table_text(
    table: pd.DataFrame,
    *,
    dataset_name: str,
    method_name: str,
    split: str,
    target_transformation: Optional[str] = None,
    float_format=None,
) -> str:
    """Return one publication-table string without writing it separately."""
    descriptor = dataset_descriptor(dataset_name)
    target = (
        f"; target representation: {target_transformation}"
        if target_transformation
        else ""
    )
    label_parts = [
        "tab",
        _latex_label_component(method_name),
        _latex_label_component(split),
        _latex_label_component(dataset_name),
    ]
    if target_transformation:
        label_parts.append(_latex_label_component(target_transformation))
    return publication_latex(
        table,
        caption=f"{descriptor}. {method_name} {split} performance{target}.",
        label=":".join(label_parts),
        float_format=float_format,
    )
