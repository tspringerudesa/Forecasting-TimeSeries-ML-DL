"""Compare verified BCRA credit perimeters in the pre-1993 activity proxy.

This is a read-only research probe.  It uses a frozen raw snapshot and prints
diagnostics; it does not write or alter a dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_dataset import (
    _archival_activity_patches,
    _fit_activity_log_level_proxy,
    _linked_emae,
    _read_indec_series,
    construct_cpi,
)


def _panser_series(path: Path, series_id: int) -> pd.Series:
    frame = pd.read_csv(
        path,
        sep=";",
        names=["series_id", "date", "value"],
        encoding="latin-1",
        dtype={"series_id": "Int64", "date": str, "value": str},
    )
    selected = frame.loc[frame["series_id"].eq(series_id)].copy()
    selected["date"] = pd.to_datetime(
        selected["date"], format="%d/%m/%Y", errors="raise"
    ).dt.to_period("M").dt.to_timestamp()
    selected["value"] = pd.to_numeric(
        selected["value"].str.replace(",", ".", regex=False), errors="raise"
    )
    return (
        selected.drop_duplicates("date", keep="last")
        .set_index("date")["value"]
        .sort_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-snapshot", required=True, type=Path)
    args = parser.parse_args()
    raw = args.raw_snapshot.resolve()

    cpi, _ = construct_cpi(raw)
    autos = _read_indec_series(raw, "330.3_PRODUCCIONLES__22") + _read_indec_series(
        raw, "330.3_PRODUCCIONSTO__16"
    )
    steel = _read_indec_series(raw, "359.3_ACERO_CRUDUDO__11")
    archival_autos, archival_steel, _ = _archival_activity_patches()
    autos.loc[archival_autos.index] = archival_autos
    steel.loc[archival_steel.index] = archival_steel
    emae, _ = _linked_emae(raw, seasonally_adjusted=False)

    code_23 = _panser_series(raw / "bcra" / "panser.txt", 23)
    candidates = {
        "code_23_private_nonfinancial_peso": code_23,
        "code_23_plus_25_private_and_public_nonfinancial_peso": (
            code_23 + _panser_series(raw / "bcra" / "panser.txt", 25)
        ),
        "code_22_total_all_currency_rejected": _panser_series(
            raw / "bcra" / "panser.txt", 22
        ),
    }
    rows: list[dict[str, object]] = []
    for name, nominal_credit in candidates.items():
        real_credit = nominal_credit / cpi["arg_cpi_level"].reindex(
            nominal_credit.index
        )
        components = pd.concat(
            {
                "log_real_credit": np.log(real_credit.where(real_credit > 0)),
                "log_auto_production": np.log(autos.where(autos > 0)),
                "log_steel_production": np.log(steel.where(steel > 0)),
            },
            axis=1,
        ).sort_index()
        try:
            fit = _fit_activity_log_level_proxy(
                components,
                emae,
                fit_start="1993-01-01",
                fit_end="2013-12-01",
            )
            predicted = fit["predicted"]
            level_holdout = pd.concat(
                {"predicted": predicted, "emae": emae}, axis=1
            ).loc["2014-01-01":"2015-12-01"].dropna()
            growth_holdout = pd.concat(
                {
                    "predicted": np.log(predicted).diff(),
                    "emae": np.log(emae).diff(),
                },
                axis=1,
            ).loc["2014-01-01":"2015-12-01"].dropna()
            rows.append(
                {
                    "candidate": name,
                    "status": "estimated",
                    "training_n": len(fit["train"]),
                    "in_sample_log_level_corr": fit["log_level_corr"],
                    "in_sample_log_growth_corr": fit["growth_corr"],
                    "holdout_log_level_corr": np.log(level_holdout).corr().iloc[0, 1],
                    "holdout_log_growth_corr": growth_holdout.corr().iloc[0, 1],
                    "pc1_loadings": list(map(float, fit["loadings"])),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "candidate": name,
                    "status": f"not estimable: {exc}",
                }
            )
    print(pd.DataFrame(rows).to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
