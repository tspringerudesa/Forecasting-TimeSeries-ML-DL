"""
Thesis Evaluation Module

Handles model evaluation, metrics calculation, and results formatting.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import os
import time
from itertools import cycle

import plotly.express as px
import plotly.graph_objects as go
from datasetsforecast.losses import mae, mse, rmse
from src.utils.ts_utils import forecast_bias_NIXTLA
from statsforecast.core import StatsForecast
from utilsforecast.evaluation import evaluate
from thesis.runtime import CPU_THREADS
try:
    from reporting import dataset_descriptor, publication_latex
except ImportError:
    from thesis.reporting import dataset_descriptor, publication_latex

# ============================================================================
# METRICS
# ============================================================================

def get_eval_metrics() -> List:
    """
    Get list of evaluation metrics.
    
    Returns:
    --------
    List
        [mae, mse, rmse]
    
    Note: forecast_bias is calculated separately due to different signature requirements.
    """
    return [mae, mse, rmse]


def get_metric_names() -> List[str]:
    """Get names of evaluation metrics."""
    return ['mae', 'mse', 'rmse', 'forecast_bias']


def prepare_comparison_metrics(
    metrics_df: pd.DataFrame,
    dataset_name: str,
    dataset_type: str,
    series_col: str = "unique_id",
    model_family: str = None,
    target_transformation: str = "Original",
    split: str = "validation",
) -> pd.DataFrame:
    """Normalize model metrics to the shared thesis comparison schema."""
    table = metrics_df.copy().rename(
        columns={
            series_col: "Series Name",
            "mae": "MAE",
            "mse": "MSE",
            "rmse": "RMSE",
            "forecast_bias": "Forecast Bias",
        }
    )
    table.insert(0, "Dataset", dataset_name)
    table.insert(1, "Dataset Type", dataset_type)
    if model_family is not None:
        table.insert(3, "Model Family", model_family)
    if "Target Transformation" not in table.columns:
        table.insert(4 if model_family is not None else 3, "Target Transformation", target_transformation)
    table.insert(5 if model_family is not None else 4, "Split", split)

    ordered = [
        "Dataset",
        "Dataset Type",
        "Series Name",
        "Model Family",
        "Model",
        "Target Transformation",
        "Configuration Source",
        "Split",
        "Validation Selection",
        "MAE",
        "MSE",
        "RMSE",
        "Forecast Bias",
        "Preprocessing Time",
        "Tuning Time",
        "Training Time",
        "Forecast Time",
        "Best Iteration",
        "Time Elapsed",
    ]
    return table[[column for column in ordered if column in table.columns]]


def export_comparison_metrics(
    metrics_df: pd.DataFrame,
    method_name: str,
    dataset_name: str,
    output_dir: str = "thesis/metrics",
) -> Dict[str, str]:
    """Write one canonical metric table as a portable CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, f"{method_name}_metrics_{dataset_name}")
    csv_path = f"{stem}.csv"
    metrics_df.to_csv(csv_path, index=False)
    return {"csv": csv_path}


def plot_forecast(
    pred_df: pd.DataFrame,
    forecast_columns: List[str],
    timestamp_col: str,
    target_col: str = "y",
    forecast_display_names: List[str] = None,
    *,
    title: str = None,
    x_axis_title: str = "Date",
    y_axis_title: str = None,
):
    """Plot actuals and forecasts using the thesis' classical-model style."""
    if forecast_display_names is None:
        forecast_display_names = forecast_columns
    else:
        assert len(forecast_columns) == len(forecast_display_names)

    mask = pred_df[forecast_columns].notna().any(axis=1)
    colors = [
        color.replace("rgb", "rgba").replace(")", ", <alpha>)")
        for color in px.colors.qualitative.Dark2
    ]
    actual_color = colors[0]
    forecast_colors = cycle(colors[1:])
    dash_types = cycle(["dash", "dot", "dashdot"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pred_df.loc[mask, timestamp_col],
            y=pred_df.loc[mask, target_col],
            mode="lines",
            line=dict(color=actual_color.replace("<alpha>", "0.3")),
            name="Original Data",
        )
    )
    for column, display_name in zip(forecast_columns, forecast_display_names):
        fig.add_trace(
            go.Scatter(
                x=pred_df.loc[mask, timestamp_col],
                y=pred_df.loc[mask, column],
                mode="lines",
                line=dict(
                    dash=next(dash_types),
                    color=next(forecast_colors).replace("<alpha>", "1"),
                ),
                name=display_name,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        margin={"l": 90, "r": 35, "t": 90 if title else 45, "b": 65},
        template="plotly_white",
    )
    return fig


def plot_forecast_matplotlib(
    pred_df: pd.DataFrame,
    forecast_columns: List[str],
    timestamp_col: str,
    target_col: str = "y",
    forecast_display_names: List[str] = None,
    *,
    title: str = None,
    x_axis_title: str = "Date",
    y_axis_title: str = None,
):
    """Create the same forecast comparison as a fast static Matplotlib figure."""
    import matplotlib.pyplot as plt

    if forecast_display_names is None:
        forecast_display_names = forecast_columns
    else:
        assert len(forecast_columns) == len(forecast_display_names)

    mask = pred_df[forecast_columns].notna().any(axis=1)
    plot_data = pred_df.loc[mask]
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        plot_data[timestamp_col],
        plot_data[target_col],
        color="#496a81",
        alpha=0.55,
        linewidth=1.6,
        label="Original Data",
    )
    forecast_colors = cycle(["#c47a5a", "#6f8f72", "#8c6f9e", "#b49a55"])
    line_styles = cycle(["--", ":", "-."])
    for column, display_name in zip(forecast_columns, forecast_display_names):
        axis.plot(
            plot_data[timestamp_col],
            plot_data[column],
            color=next(forecast_colors),
            linestyle=next(line_styles),
            linewidth=1.8,
            label=display_name,
        )
    axis.set(
        title=title,
        xlabel=x_axis_title,
        ylabel=y_axis_title,
    )
    axis.grid(alpha=0.20)
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig


def build_results_table(
    metrics: pd.DataFrame,
    id_col: str,
    dataset_type: str,
) -> pd.DataFrame:
    """Build the seven-column results table used by classical_models.ipynb."""
    if dataset_type == "synthetic":
        metrics_copy = metrics.copy()
        metrics_copy["series_type"] = metrics_copy[id_col].str.rsplit("_", n=1).str[0]
        result_table = (
            metrics_copy.groupby(["series_type", "Model"])
            .agg(
                {
                    "mae": "mean",
                    "mse": "mean",
                    "rmse": "mean",
                    "forecast_bias": "mean",
                    "Time Elapsed": "mean",
                }
            )
            .reset_index()
        )
        result_table.columns = [
            "Series Name",
            "Model",
            "MAE",
            "MSE",
            "RMSE",
            "Forecast Bias",
            "Time Elapsed",
        ]
    else:
        columns = [
            id_col,
            "Model",
            "mae",
            "mse",
            "rmse",
            "forecast_bias",
            "Time Elapsed",
        ]
        result_table = metrics[[column for column in columns if column in metrics]].copy()
        result_table.columns = [
            "Series Name",
            "Model",
            "MAE",
            "MSE",
            "RMSE",
            "Forecast Bias",
            "Time Elapsed",
        ][: len(result_table.columns)]

    return result_table.sort_values(["Series Name", "MAE"]).reset_index(drop=True)


def style_results_table(df: pd.DataFrame):
    """Format and highlight a results table exactly as in the classical analysis."""
    formats = {
        "MAE": "{:.4f}",
        "MSE": "{:.4f}",
        "RMSE": "{:.4f}",
        "Forecast Bias": "{:.2f}",
        "Time Elapsed": "{:.6f}",
    }
    styled = df.style.format({key: value for key, value in formats.items() if key in df})
    if "Series Name" in df:
        groups = [
            df.index[df["Series Name"].eq(series_name)]
            for series_name in df["Series Name"].drop_duplicates()
        ]
    else:
        groups = [df.index]

    for metric in ["MAE", "MSE", "RMSE", "Time Elapsed"]:
        if metric not in df:
            continue
        for group_index in groups:
            series_data = df.loc[group_index]
            best_index = series_data[metric].idxmin()
            styled = styled.highlight_max(
                subset=pd.IndexSlice[[best_index], [metric]], color="lightgreen"
            )

    if "Forecast Bias" in df:
        for group_index in groups:
            series_data = df.loc[group_index]
            best_index = series_data["Forecast Bias"].abs().idxmin()
            styled = styled.highlight_max(
                subset=pd.IndexSlice[[best_index], ["Forecast Bias"]],
                color="lightgreen",
            )
    return styled


def export_results_table_latex(
    result_table: pd.DataFrame,
    dataset_name: str,
    method_name: str,
    output_dir: str = "thesis/results",
) -> str:
    """Export a result table using the same tabular style as the classical files."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"results_table_{method_name}_{dataset_name}.tex"
    )
    latex_table = publication_latex(
        result_table,
        caption=f"{dataset_descriptor(dataset_name)}. {method_name.replace('_', ' ')} results.",
        label=f"tab:{method_name.replace('_', '-')}-{dataset_name.replace('_', '-')}",
        float_format=lambda value: (
            f"{value:.6f}" if abs(value) < 100 else f"{value:.2f}"
        ),
    )
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(latex_table)
    return output_file


def export_results_tables_latex(
    result_tables: Dict[str, pd.DataFrame],
    dataset_name: str,
    method_name: str,
    output_dir: str = "thesis/results",
) -> str:
    """Export multiple target-version tables into one LaTeX file."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"results_table_{method_name}_{dataset_name}.tex"
    )
    sections = []
    for target_version, result_table in result_tables.items():
        latex_table = publication_latex(
            result_table,
            caption=(
                f"{dataset_descriptor(dataset_name)}. "
                f"{method_name.replace('_', ' ')} results on the {target_version} target."
            ),
            label=(
                f"tab:{method_name.replace('_', '-')}-"
                f"{dataset_name.replace('_', '-')}-{target_version.lower()}"
            ),
            float_format=lambda value: (
                f"{value:.6f}" if abs(value) < 100 else f"{value:.2f}"
            ),
        )
        sections.append(f"% {target_version} target\n{latex_table}")
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n\n".join(sections))
    return output_file


# ============================================================================
# PERFORMANCE EVALUATION
# ============================================================================

def evaluate_performance(
    ts_train: pd.DataFrame,
    ts_test: pd.DataFrame,
    models: List,
    metrics: List,
    freq: str,
    id_col: str = 'unique_id',
    time_col: str = 'ds',
    target_col: str = 'y',
    h: int = None,
    print_output: bool = True,
    return_fitted_models: bool = True,
    n_jobs: int = CPU_THREADS,
) -> Tuple[pd.DataFrame, pd.DataFrame, StatsForecast]:
    """
    Evaluate model performance on time series data.
    
    Parameters:
    -----------
    ts_train : pd.DataFrame
        Training data with columns [id_col, time_col, target_col]
    ts_test : pd.DataFrame
        Test/validation data with columns [id_col, time_col, target_col]
    models : List
        List of instantiated models to evaluate
    metrics : List
        List of metric functions
    freq : str
        Frequency string (e.g., 'D', 'MS')
    id_col : str
        Column name for series ID
    time_col : str
        Column name for time index
    target_col : str
        Column name for target variable
    h : int
        Forecast horizon (defaults to len(ts_test))
    print_output : bool
        Whether to print progress
    return_fitted_models : bool
        Whether to return the fitted StatsForecast object
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, StatsForecast] or Tuple[pd.DataFrame, pd.DataFrame]
        (results_df, metrics_df, fitted_sf) or (results_df, metrics_df)
        - results_df: forecasts with columns [unique_id, ds, y, Model_1, Model_2, ...]
        - metrics_df: evaluation metrics with columns [unique_id, Model, metric_1, metric_2, ...]
        - fitted_sf: StatsForecast object with fitted models (optional)
    """
    if h is None:
        h = len(ts_test)
    
    if print_output:
        print(f"Evaluating {len(models)} models...")
        print(f"  Horizon: {h} steps")
        print(f"  Metrics: {[m.__name__ for m in metrics]}")
    
    # Time at model level: instantiation + forecast per model
    timing = {}
    all_forecasts = None
    
    for model in models:
        model_name = model.__class__.__name__
        
        # Start timer for this model
        start_time = time.time()
        
        # Instantiate StatsForecast with this model
        sf = StatsForecast(
            models=[model],
            freq=freq,
            n_jobs=n_jobs,
        )
        
        # Generate forecasts
        forecasts = sf.forecast(
            df=ts_train,
            h=h,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col
        )
        
        # Record elapsed time for this model
        duration = time.time() - start_time
        timing[model_name] = duration
        
        # Merge forecasts with actuals
        forecasts = forecasts.reset_index()
        
        # Accumulate forecasts from all models
        if all_forecasts is None:
            all_forecasts = forecasts.copy()
        else:
            # Merge only the forecast column from this model
            all_forecasts = all_forecasts.merge(
                forecasts[[id_col, time_col, model_name]],
                on=[id_col, time_col],
                how='left'
            )
    
    # Merge test actuals once
    ts_test_reset = ts_test.reset_index() if hasattr(ts_test.index, 'name') else ts_test.copy()
    
    # Ensure time column is datetime
    if time_col in all_forecasts.columns:
        all_forecasts[time_col] = pd.to_datetime(all_forecasts[time_col])
    if time_col in ts_test_reset.columns:
        ts_test_reset[time_col] = pd.to_datetime(ts_test_reset[time_col])
    
    # Merge forecasts with actual values
    results = all_forecasts.merge(
        ts_test_reset[[id_col, time_col, target_col]],
        on=[id_col, time_col],
        how='left'
    )
    
    # Get forecast column names (all columns except id, time, target, and index)
    forecast_cols = [col for col in results.columns if col not in [id_col, time_col, target_col] and col != 'index']
    
    # Calculate metrics manually for each model and series
    metrics_list = []
    for model_col in forecast_cols:
        model_name = model_col
        for _, group in results.groupby(id_col):
            y_true = group[target_col].values
            y_pred = group[model_col].values
            
            metric_dict = {id_col: group[id_col].iloc[0], 'Model': model_name}
            for metric_fn in metrics:
                metric_name = metric_fn.__name__
                try:
                    metric_value = metric_fn(y_true, y_pred)
                    metric_dict[metric_name] = metric_value
                except Exception as e:
                    metric_dict[metric_name] = np.nan
            
            # Use model-level timing
            metric_dict['Time Elapsed'] = timing.get(model_name, 0)
            metrics_list.append(metric_dict)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Calculate forecast_bias separately and add to metrics_df
    try:
        bias_df = forecast_bias_NIXTLA(
            results,
            models=forecast_cols,
            id_col=id_col,
            target_col=target_col
        )
        
        # Melt bias_df to long format and add to metrics_df
        bias_df_long = bias_df.melt(id_vars=[id_col], var_name='Model', value_name='forecast_bias')
        metrics_df = metrics_df.merge(bias_df_long, on=[id_col, 'Model'], how='left')
    except Exception as e:
        if print_output:
            print(f"⚠ Could not calculate forecast_bias: {str(e)[:80]}")
        # Add empty forecast_bias column if calculation fails
        metrics_df['forecast_bias'] = np.nan
    
    if print_output:
        print(f"✓ Evaluation complete")
        print(f"  Forecasts: {len(results)} rows")
        print(f"  Metrics: {len(metrics_df)} rows")
    
    if return_fitted_models:
        return results, metrics_df, sf
    else:
        return results, metrics_df


def evaluate_transformed_performance(
    ts_train: pd.DataFrame,
    ts_test: pd.DataFrame,
    models: List,
    metrics: List,
    transformer_factory,
    freq: str,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    h: int = None,
    model_suffix: str = "_AutoStationary",
    transformation_name: str = "AutoStationary",
    print_output: bool = True,
    n_jobs: int = CPU_THREADS,
):
    """Forecast a training-only transformed target and score on original scale."""
    transformed_parts = []
    transformers = {}

    for series_id, group in ts_train.groupby(id_col, sort=False):
        group = group.sort_values(time_col)
        index = pd.DatetimeIndex(pd.to_datetime(group[time_col]))
        original = pd.Series(
            group[target_col].astype(float).to_numpy(),
            index=index,
            name=target_col,
        )
        transformer = transformer_factory()
        transformed = transformer.fit_transform(original, freq=freq)
        if len(transformed) != len(group):
            raise ValueError(
                "Target transformer must preserve length for StatsForecast evaluation"
            )
        transformed_parts.append(
            pd.DataFrame(
                {
                    id_col: series_id,
                    time_col: index,
                    target_col: np.asarray(transformed).reshape(-1),
                }
            )
        )
        transformers[series_id] = transformer

    transformed_train = pd.concat(transformed_parts, ignore_index=True)
    transformed_results, transformed_metrics, fitted_sf = evaluate_performance(
        ts_train=transformed_train,
        ts_test=ts_test,
        models=models,
        metrics=metrics,
        freq=freq,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        h=h,
        print_output=print_output,
        return_fitted_models=True,
        n_jobs=n_jobs,
    )

    base_model_names = [model.__class__.__name__ for model in models]
    results = transformed_results[[id_col, time_col, target_col]].copy()
    metric_rows = []

    for series_id, group in transformed_results.groupby(id_col, sort=False):
        group = group.sort_values(time_col)
        transformer = transformers[series_id]
        prediction_index = pd.DatetimeIndex(pd.to_datetime(group[time_col]))
        actual = group[target_col].astype(float).to_numpy()

        for model_name in base_model_names:
            output_name = f"{model_name}{model_suffix}"
            transformed_prediction = pd.Series(
                group[model_name].astype(float).to_numpy(),
                index=prediction_index,
                name=output_name,
            )
            prediction = np.asarray(
                transformer.inverse_transform(transformed_prediction)
            ).reshape(-1)
            result_mask = results[id_col].eq(series_id)
            result_index = results.loc[result_mask].sort_values(time_col).index
            results.loc[result_index, output_name] = prediction

            timing_row = transformed_metrics.loc[
                transformed_metrics[id_col].eq(series_id)
                & transformed_metrics["Model"].eq(model_name),
                "Time Elapsed",
            ]
            metric_row = {
                id_col: series_id,
                "Model": output_name,
                "Base Model": model_name,
                "Target Transformation": transformation_name,
                "Time Elapsed": timing_row.iloc[0] if not timing_row.empty else np.nan,
            }
            for metric_fn in metrics:
                metric_row[metric_fn.__name__] = metric_fn(actual, prediction)

            bias_frame = pd.DataFrame(
                {id_col: series_id, target_col: actual, output_name: prediction}
            )
            metric_row["forecast_bias"] = forecast_bias_NIXTLA(
                bias_frame,
                models=[output_name],
                id_col=id_col,
                target_col=target_col,
            )[output_name].iloc[0]
            metric_rows.append(metric_row)

    return (
        results.sort_values([id_col, time_col]).reset_index(drop=True),
        pd.DataFrame(metric_rows),
        fitted_sf,
        transformers,
    )


# ============================================================================
# RESULTS FORMATTING & DISPLAY
# ============================================================================

def color_best_per_series(series_data: pd.DataFrame, metric: str = 'mae', ascending: bool = True) -> List[str]:
    """
    Apply green background to best performing model per series.
    
    Handles special case for Forecast Bias (closest to zero).
    
    Parameters:
    -----------
    series_data : pd.DataFrame
        Data for one series
    metric : str
        Metric column to highlight
    ascending : bool
        Whether lower values are better
        
    Returns:
    --------
    List[str]
        CSS styling for each row
    """
    styles = [''] * len(series_data)
    
    if metric not in series_data.columns:
        return styles
    
    if metric == 'forecast_bias':
        # For Forecast Bias: closest to zero is best
        best_idx = series_data[metric].abs().idxmin()
    else:
        # For MAE/MSE/RMSE: lower is better
        if ascending:
            best_idx = series_data[metric].idxmin()
        else:
            best_idx = series_data[metric].idxmax()
    
    if best_idx is not None:
        styles[list(series_data.index).index(best_idx)] = 'background-color: lightgreen'
    
    return styles


def format_metrics_table(
    metrics_df: pd.DataFrame,
    id_col: str = 'unique_id',
    dataset_type: str = 'manual',
    highlight_metric: str = 'mae'
) -> pd.DataFrame:
    """
    Format and pivot metrics table for display.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        Raw metrics from evaluate_performance
    id_col : str
        Column name for series ID
    dataset_type : str
        Type of dataset ('manual' or 'synthetic')
    highlight_metric : str
        Metric to highlight best values
        
    Returns:
    --------
    pd.DataFrame
        Formatted table pivoted by model
    """
    if dataset_type == 'manual':
        # Group by series, average across all metrics
        result_table = metrics_df.groupby(id_col).mean(numeric_only=True)
    else:
        # For synthetic: group by series type
        metrics_df['series_type'] = (
            metrics_df[id_col].astype(str).str.rsplit('_', n=1).str[0]
        )
        result_table = metrics_df.groupby('series_type').mean(numeric_only=True)
    
    # Remove 'Model' column if present (it's the index now)
    if 'Model' in result_table.columns:
        result_table = result_table.drop('Model', axis=1)
    
    # Round to 4 decimal places
    result_table = result_table.round(4)
    
    return result_table


def export_latex_table(
    result_table: pd.DataFrame,
    dataset_name: str,
    output_dir: str = 'thesis/results',
    caption: str = None,
    label: str = None
) -> str:
    """
    Export metrics table to LaTeX format.
    
    Parameters:
    -----------
    result_table : pd.DataFrame
        Formatted metrics table
    dataset_name : str
        Name of dataset (used in filename)
    output_dir : str
        Directory to save LaTeX file
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
        
    Returns:
    --------
    str
        Path to generated LaTeX file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/results_table_{dataset_name}.tex"
    
    export_table = result_table.reset_index()
    latex_code = publication_latex(
        export_table,
        caption=caption or (
            f"{dataset_descriptor(dataset_name)}. Forecast performance."
        ),
        label=label or f"tab:forecast-{dataset_name.replace('_', '-')}",
        float_format=lambda value: f"{value:.4f}",
    )
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    return output_file
