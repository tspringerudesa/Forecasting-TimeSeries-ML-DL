"""Isolated worker for rolling exogenous neural ShapTime attribution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

THESIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THESIS_DIR.parent
for search_path in (REPO_ROOT, THESIS_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--probe", action="store_true")
    return parser.parse_args()


def _probe() -> None:
    # Import only the numerical stack used by the worker. In particular, do not
    # import SHAP or LightGBM, which may initialize LLVM OpenMP in the notebook.
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import torch
    import neuralforecast  # noqa: F401

    print(
        "isolated neural worker ready "
        f"(torch={torch.__version__}, cuda={torch.cuda.is_available()})",
        flush=True,
    )


def main() -> None:
    args = _arguments()
    if args.probe:
        _probe()
        return
    if args.request is None:
        raise ValueError("--request is required unless --probe is used")

    # Keep native numerical libraries conservative in this short-lived worker.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import pandas as pd

    from exogenous_evaluation import rolling_dl_shaptime

    request = json.loads(args.request.read_text(encoding="utf-8"))
    fit_df = pd.read_csv(request["fit_path"], parse_dates=["date"])
    eval_df = pd.read_csv(request["eval_path"], parse_dates=["date"])
    output_path = Path(request["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"{request['model_name']}: fitting one h=1 model and explaining "
        f"{len(eval_df)} rolling origins",
        flush=True,
    )
    attribution = rolling_dl_shaptime(
        fit_df=fit_df,
        eval_df=eval_df,
        feature_columns=request["feature_columns"],
        model_name=request["model_name"],
        resolved_config=request["resolved_config"],
        feature_mode=request["feature_mode"],
        n_super_times=request["n_super_times"],
        seasonal_background_lag=request["seasonal_background_lag"],
        expected_predictions=request["expected_predictions"],
        require_prediction_match=bool(
            request.get("require_prediction_match", False)
        ),
        show_progress=True,
    )
    temporary_output = output_path.with_suffix(output_path.suffix + ".partial")
    attribution.to_csv(temporary_output, index=False)
    temporary_output.replace(output_path)
    print(
        f"saved {len(attribution)} origin-window attributions to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
