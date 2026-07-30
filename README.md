# Forecasting Time Series with Classical, Machine Learning, and Deep Learning Models

This repository contains the paper and source code for my undergraduate thesis, *Back to Which Future? Multiple Approaches to Time Series Forecasting: from Classical Statistical Methods to Deep Learning*, submitted at Universidad de San Andrés in July 2026.

[Read the thesis](paper.pdf)

The project compares classical statistical methods, tree-based machine-learning models, and neural forecasting architectures across macroeconomic and synthetic time series. It also studies global forecasting, forecastability, target transformations, exogenous-variable forecasting, model interpretation, and adaptive conformal prediction intervals.

## Repository scope

This is a compact publication of the thesis rather than a fully self-contained reproduction archive. It includes:

- the final thesis PDF;
- the Jupyter notebooks used for the experiments and methodological figures;
- the thesis-specific Python modules used by those notebooks.

To keep the repository manageable and avoid republishing data from other sources, it excludes datasets, trained artifacts, saved configurations, cached predictions, generated tables and figures, and LaTeX build files. Those outputs can be regenerated where the underlying data sources remain available.

## Required parent repository

The code extends the companion repository for *Modern Time Series Forecasting with Python, Second Edition* and imports utilities from its `src` package. Clone that repository first, then clone this repository into it under the directory name `thesis`:

```powershell
git clone https://github.com/PacktPublishing/Modern-Time-Series-Forecasting-with-Python-2E.git
cd Modern-Time-Series-Forecasting-with-Python-2E
git clone https://github.com/tspringerudesa/Forecasting-TimeSeries-ML-DL.git thesis
```

The resulting layout should be:

```text
Modern-Time-Series-Forecasting-with-Python-2E/
├── src/
├── notebooks/
└── thesis/
```

Follow the upstream repository's environment instructions before running the notebooks. Some experiments are computationally expensive and benefit substantially from a CUDA-capable GPU. Data downloaded from FRED requires a `FRED_API_KEY` environment variable; credentials are not included here.

## Main notebooks

- `data_wrangling.ipynb`: preparation and exploration of the manual datasets
- `synthethic_ts_generation.ipynb`: synthetic data-generating processes
- `forecastability_analysis.ipynb`: forecastability diagnostics
- `classical_models.ipynb`: local classical forecasts
- `ml_models.ipynb`: local tree-based forecasts and blending
- `dl_models.ipynb`: local deep-learning forecasts
- `global_models.ipynb`: global forecasting across related synthetic series
- `predictable_series_conformal.ipynb`: conformal forecasts for predictable series
- `sp500_probabilistic_forecasting.ipynb`: conformal forecasts for S&P 500 returns
- `exogenous_models.ipynb`: exogenous forecasting and the inflation replication exercise
- `hyperparameter_search_comparison.ipynb`: search-method comparison
- `methodology_figures.ipynb`: figures used to explain the methodology

The notebooks preserve their original experimental outputs where present, but reproducing every result requires obtaining or regenerating the omitted data and intermediate artifacts.

## Attribution

This work builds on the code and examples accompanying *Modern Time Series Forecasting with Python, Second Edition*. The upstream project and its `src` package remain the work of their respective authors and publisher.

## Citation

```bibtex
@thesis{springerurbieta2026forecasting,
  author = {Springer Urbieta, Teodoro},
  title = {Back to Which Future? Multiple Approaches to Time Series Forecasting: from Classical Statistical Methods to Deep Learning},
  school = {Universidad de San Andrés},
  year = {2026},
  type = {Undergraduate thesis}
}
```
