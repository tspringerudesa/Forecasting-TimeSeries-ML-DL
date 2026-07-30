from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from pipeline.build_dataset import VARIABLE_COLUMNS
from pipeline.common import (
    METADATA_ROOT,
    PROJECT_ROOT,
    TARGET_END,
    TARGET_START,
    read_json,
    safe_relpath,
    sha256_file,
    utc_now_iso,
    write_json,
)


def _check(
    rows: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    observed: Any,
    expected: Any,
    severity: str = "error",
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "passed": bool(passed),
            "severity": severity,
            "observed": str(observed),
            "expected": str(expected),
        }
    )


def _coverage_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variable_id, column in VARIABLE_COLUMNS.items():
        series = frame[column]
        nonmissing = series.dropna()
        rows.append(
            {
                "variable_id": variable_id,
                "column_name": column,
                "first_nonmissing": (
                    nonmissing.index.min().date().isoformat() if len(nonmissing) else ""
                ),
                "last_nonmissing": (
                    nonmissing.index.max().date().isoformat() if len(nonmissing) else ""
                ),
                "nonmissing_months": int(series.notna().sum()),
                "missing_months": int(series.isna().sum()),
                "coverage_pct": float(100.0 * series.notna().mean()),
                "minimum": float(nonmissing.min()) if len(nonmissing) else np.nan,
                "maximum": float(nonmissing.max()) if len(nonmissing) else np.nan,
            }
        )
    return rows


def validate_latest_build(snapshot_id: str | None = None) -> dict[str, Any]:
    pointer = read_json(METADATA_ROOT / "latest_build.json")
    selected = snapshot_id or str(pointer["snapshot_id"])
    if selected != str(pointer["snapshot_id"]):
        output = PROJECT_ROOT / "data" / "processed" / "snapshots" / selected
    else:
        output = PROJECT_ROOT / pointer["output_directory"]
    wide_path = output / "forte_13_public_proxy_monthly_1965_2024.csv"
    details_path = output / "forte_13_construction_details_monthly.csv"
    long_path = output / "forte_13_public_proxy_long_with_availability.csv"
    metrics_path = output / "construction_metrics.json"
    source_map_path = output / "audits" / "variable_raw_source_map.csv"
    wide = pd.read_csv(wide_path, parse_dates=["date"]).set_index("date")
    details = pd.read_csv(details_path, parse_dates=["date"]).set_index("date")
    long = pd.read_csv(long_path, dtype={"variable_id": str})
    metrics = read_json(metrics_path)
    source_map = pd.read_csv(source_map_path, dtype=str).fillna("")
    expected_index = pd.date_range(TARGET_START, TARGET_END, freq="MS", name="date")
    checks: list[dict[str, Any]] = []

    _check(checks, "row_count", len(wide) == len(expected_index), len(wide), len(expected_index))
    _check(
        checks,
        "monthly_index_exact",
        wide.index.equals(expected_index),
        f"{wide.index.min()}..{wide.index.max()} ({wide.index.nunique()} unique)",
        f"{expected_index.min()}..{expected_index.max()} ({len(expected_index)} unique)",
    )
    _check(
        checks,
        "thirteen_columns",
        list(wide.columns) == list(VARIABLE_COLUMNS.values()),
        list(wide.columns),
        list(VARIABLE_COLUMNS.values()),
    )
    _check(
        checks,
        "no_infinite_values",
        not np.isinf(wide.select_dtypes(include=[np.number]).to_numpy()).any(),
        int(np.isinf(wide.select_dtypes(include=[np.number]).to_numpy()).sum()),
        0,
    )
    _check(
        checks,
        "long_row_count",
        len(long) == len(wide) * 13,
        len(long),
        len(wide) * 13,
    )
    _check(
        checks,
        "long_unique_keys",
        not long.duplicated(["date", "variable_id"]).any(),
        int(long.duplicated(["date", "variable_id"]).sum()),
        0,
    )

    # Formula checks use the transparent construction-detail file.
    cpi_recomputed = 100.0 * details["cpi__arg_cpi_level"].pct_change(fill_method=None)
    cpi_diff = (
        cpi_recomputed - wide[VARIABLE_COLUMNS["U01"]]
    ).abs().dropna()
    _check(
        checks,
        "cpi_growth_formula",
        bool((cpi_diff < 1e-10).all()),
        float(cpi_diff.max()) if len(cpi_diff) else np.nan,
        "<1e-10 percentage point",
    )
    gap_recomputed = 100.0 * (
        details["parallel_fx__parallel_fx_level"]
        / details["parallel_fx__official_fx_level"]
        - 1.0
    )
    gap_diff = (gap_recomputed - wide[VARIABLE_COLUMNS["U12"]]).abs().dropna()
    _check(
        checks,
        "fx_gap_formula",
        bool((gap_diff < 1e-10).all()),
        float(gap_diff.max()) if len(gap_diff) else np.nan,
        "<1e-10 percentage point",
    )
    nir_recomputed = (
        details["nir__nir_strict_nominal_musd"]
        * details["nir__usd_jul2024_deflator"]
    )
    nir_diff = (nir_recomputed - wide[VARIABLE_COLUMNS["U13"]]).abs().dropna()
    _check(
        checks,
        "nir_deflation_formula",
        bool((nir_diff < 1e-8).all()),
        float(nir_diff.max()) if len(nir_diff) else np.nan,
        "<1e-8 million USD",
    )
    july_nir = float(details.loc["2024-07-01", "nir__nir_strict_nominal_musd"])
    _check(
        checks,
        "nir_july2024_public_formula",
        abs(july_nir - (-6493.108912)) < 0.02,
        july_nir,
        "-6493.108912 +/- 0.02 million USD (PDF component rounding)",
    )

    activity_gap = wide.loc["1989-01-01":"1990-12-01", VARIABLE_COLUMNS["U03"]]
    _check(
        checks,
        "activity_gap_not_fabricated",
        bool(activity_gap.isna().all()),
        int(activity_gap.isna().sum()),
        24,
    )
    expected_activity_missing = pd.date_range(
        "1965-01-01", "1965-03-01", freq="MS"
    ).append(
        pd.date_range("1989-01-01", "1991-03-01", freq="MS")
    )
    observed_activity_missing = wide.index[
        wide[VARIABLE_COLUMNS["U03"]].isna()
    ]
    _check(
        checks,
        "activity_exact_missing_pattern",
        observed_activity_missing.equals(expected_activity_missing),
        observed_activity_missing.strftime("%Y-%m").tolist(),
        expected_activity_missing.strftime("%Y-%m").tolist(),
    )
    activity_level = details["activity__activity_level"]
    activity_growth_recomputed = 100.0 * activity_level.pct_change(fill_method=None)
    activity_growth_stored = details["activity__activity_growth_pct"]
    activity_growth_difference = (
        activity_growth_recomputed - activity_growth_stored
    ).abs().dropna()
    _check(
        checks,
        "activity_level_to_simple_growth_formula",
        (
            activity_growth_recomputed.isna().equals(
                activity_growth_stored.isna()
            )
            and bool((activity_growth_difference < 1e-10).all())
        ),
        {
            "max_abs_difference": (
                float(activity_growth_difference.max())
                if len(activity_growth_difference)
                else np.nan
            ),
            "na_masks_equal": activity_growth_recomputed.isna().equals(
                activity_growth_stored.isna()
            ),
        },
        "identical NA mask and <1e-10 percentage-point difference",
    )
    activity_3mma = activity_growth_recomputed.rolling(3, min_periods=3).mean()
    activity_formula_difference = (
        activity_3mma - wide[VARIABLE_COLUMNS["U03"]]
    ).abs().dropna()
    _check(
        checks,
        "activity_growth_to_trailing_3mma_formula",
        (
            activity_3mma.isna().equals(
                wide[VARIABLE_COLUMNS["U03"]].isna()
            )
            and bool((activity_formula_difference < 1e-10).all())
        ),
        {
            "max_abs_difference": (
                float(activity_formula_difference.max())
                if len(activity_formula_difference)
                else np.nan
            ),
            "na_masks_equal": activity_3mma.isna().equals(
                wide[VARIABLE_COLUMNS["U03"]].isna()
            ),
        },
        "identical NA mask and <1e-10 percentage-point difference",
    )
    api_patch_index = pd.date_range("1991-01-01", "1992-12-01", freq="MS")
    _check(
        checks,
        "activity_api_holes_precede_archival_patch",
        (
            details[
                "activity__auto_production_api_before_archival_patch_units"
            ].reindex(api_patch_index).isna().all()
            and details[
                "activity__steel_production_api_before_archival_patch_thousand_tonnes"
            ].reindex(api_patch_index).isna().all()
        ),
        {
            "auto_nonmissing": int(
                details[
                    "activity__auto_production_api_before_archival_patch_units"
                ].reindex(api_patch_index).notna().sum()
            ),
            "steel_nonmissing": int(
                details[
                    "activity__steel_production_api_before_archival_patch_thousand_tonnes"
                ].reindex(api_patch_index).notna().sum()
            ),
        },
        {"auto_nonmissing": 0, "steel_nonmissing": 0},
    )
    api_file_checks: dict[str, dict[str, Any]] = {}
    for name, api_column, file_column in (
        (
            "automobiles",
            "activity__auto_production_api_before_archival_patch_units",
            "activity__auto_production_official_file_units",
        ),
        (
            "steel",
            "activity__steel_production_api_before_archival_patch_thousand_tonnes",
            "activity__steel_production_official_file_thousand_tonnes",
        ),
    ):
        api_values = details[api_column]
        file_values = details[file_column]
        difference = (api_values - file_values).abs().dropna()
        api_file_checks[name] = {
            "na_masks_equal": api_values.isna().equals(file_values.isna()),
            "max_abs_difference": (
                float(difference.max()) if len(difference) else np.nan
            ),
        }
    _check(
        checks,
        "activity_api_equals_archived_official_csv",
        all(
            item["na_masks_equal"] and item["max_abs_difference"] < 1e-9
            for item in api_file_checks.values()
        ),
        api_file_checks,
        "identical NA masks and <1e-9 unit difference",
    )
    archival_sums = {
        "autos_1991": float(
            details.loc["1991", "activity__auto_production_units"].sum()
        ),
        "autos_1992": float(
            details.loc["1992", "activity__auto_production_units"].sum()
        ),
        "steel_1991": float(
            details.loc[
                "1991", "activity__steel_production_thousand_tonnes"
            ].sum()
        ),
        "steel_1992": float(
            details.loc[
                "1992", "activity__steel_production_thousand_tonnes"
            ].sum()
        ),
    }
    _check(
        checks,
        "activity_archival_annual_sums",
        all(
            abs(archival_sums[key] - expected) < 0.01
            for key, expected in {
                "autos_1991": 138958.0,
                "autos_1992": 262022.0,
                "steel_1991": 2972.0,
                "steel_1992": 2679.9,
            }.items()
        ),
        archival_sums,
        {
            "autos_1991": 138958.0,
            "autos_1992": 262022.0,
            "steel_1991": 2972.0,
            "steel_1992": 2679.9,
        },
    )
    activity_metrics = metrics["activity"]
    predicted_activity = details["activity__activity_pca_predicted_level"]
    emae_activity = details["activity__emae_linked_original"]
    in_sample_levels = pd.concat(
        {
            "predicted": np.log(predicted_activity.where(predicted_activity > 0)),
            "emae": np.log(emae_activity.where(emae_activity > 0)),
        },
        axis=1,
    ).loc["1993-01-01":"2013-12-01"].dropna()
    in_sample_growth = in_sample_levels.diff().dropna()
    holdout_levels = pd.concat(
        {
            "predicted": np.log(predicted_activity.where(predicted_activity > 0)),
            "emae": np.log(emae_activity.where(emae_activity > 0)),
        },
        axis=1,
    ).loc["2014-01-01":"2015-12-01"].dropna()
    holdout_growth = pd.concat(
        {
            "predicted": np.log(predicted_activity.where(predicted_activity > 0)).diff(),
            "emae": np.log(emae_activity.where(emae_activity > 0)).diff(),
        },
        axis=1,
    ).loc["2014-01-01":"2015-12-01"].dropna()
    recomputed_correlations = {
        "in_sample_level": float(in_sample_levels.corr().iloc[0, 1]),
        "in_sample_growth": float(in_sample_growth.corr().iloc[0, 1]),
        "holdout_level": float(holdout_levels.corr().iloc[0, 1]),
        "holdout_growth": float(holdout_growth.corr().iloc[0, 1]),
    }
    metric_correlations = {
        "in_sample_level": float(
            activity_metrics["level_proxy_log_level_corr_with_emae_1993_2013"]
        ),
        "in_sample_growth": float(
            activity_metrics["level_proxy_growth_corr_with_emae_growth_1993_2013"]
        ),
        "holdout_level": float(
            activity_metrics["postfit_holdout_log_level_corr_2014_2015"]
        ),
        "holdout_growth": float(
            activity_metrics["postfit_holdout_growth_corr_2014_2015"]
        ),
    }
    _check(
        checks,
        "activity_correlations_independently_recomputed",
        (
            int(activity_metrics["pca_training_n"]) == 252
            and activity_metrics["pca_training_start"] == "1993-01-01"
            and activity_metrics["pca_training_end"] == "2013-12-01"
            and activity_metrics["pca_missing_training_months"] == []
            and max(
                abs(recomputed_correlations[key] - metric_correlations[key])
                for key in recomputed_correlations
            )
            < 1e-12
            and recomputed_correlations["in_sample_level"] > 0.8
        ),
        {
            "training_n": activity_metrics["pca_training_n"],
            "recomputed": recomputed_correlations,
            "stored_metrics": metric_correlations,
        },
        "252 complete 1993-01..2013-12 months, metric match <1e-12, in-sample level corr >0.8",
    )
    _check(
        checks,
        "activity_holdout_strength_diagnostic",
        recomputed_correlations["holdout_level"] >= 0.8,
        recomputed_correlations,
        "paper-like >0.8 correlation outside the fitted mapping window",
        severity="warning",
    )
    jan_1993_anchor_difference = abs(
        float(predicted_activity.loc["1993-01-01"])
        - float(emae_activity.loc["1993-01-01"])
    )
    loadings = pd.read_csv(output / "audits" / "activity_pca_loadings.csv")
    primary_loadings = loadings["pc1_loading_primary"].to_numpy(dtype=float)
    _check(
        checks,
        "activity_pca_anchor_and_loading_normalization",
        (
            jan_1993_anchor_difference < 1e-10
            and abs(float(np.linalg.norm(primary_loadings)) - 1.0) < 1e-12
            and float(primary_loadings.sum()) > 0
        ),
        {
            "jan1993_level_difference": jan_1993_anchor_difference,
            "loading_norm": float(np.linalg.norm(primary_loadings)),
            "loading_sum": float(primary_loadings.sum()),
        },
        "Jan-1993 anchor equal, unit-norm loadings, positive orientation",
    )
    _check(
        checks,
        "activity_1991_steel_printed_total_reconciliation",
        (
            abs(
                float(
                    activity_metrics[
                        "archival_steel_1991_monthly_sum_minus_printed_total"
                    ]
                )
                - (-0.3)
            )
            < 1e-9
        ),
        activity_metrics[
            "archival_steel_1991_monthly_sum_minus_printed_total"
        ],
        "-0.3 thousand tonnes (rounded monthly rows versus printed annual total)",
    )
    cpi_intervention = wide.loc["2007-01-01":"2015-12-01", VARIABLE_COLUMNS["U01"]]
    _check(
        checks,
        "cpi_intervention_complete",
        bool(cpi_intervention.notna().all()),
        int(cpi_intervention.notna().sum()),
        108,
    )
    wages = wide.loc["1965-02-01":, VARIABLE_COLUMNS["U02"]]
    _check(
        checks,
        "wage_splice_complete_from_1965_02",
        bool(wages.notna().all()),
        int(wages.isna().sum()),
        0,
    )
    nedd_target = details.loc[
        "1999-12-01":"2024-07-01", "nir__nir_strict_nominal_musd"
    ]
    missing_nedd = nedd_target.index[nedd_target.isna()].strftime("%Y-%m").tolist()
    _check(
        checks,
        "nedd_296_constructed_no_month_gap",
        len(nedd_target) == 296 and missing_nedd == [],
        f"rows={len(nedd_target)}, missing={missing_nedd}",
        "296 target rows; no missing months after archived-BCRA and IMF recovery",
    )
    may_2000_nir = float(
        details.loc["2000-05-01", "nir__nir_strict_nominal_musd"]
    )
    _check(
        checks,
        "nedd_may2000_archived_bcra_value",
        abs(may_2000_nir - 23784.0) < 0.01,
        may_2000_nir,
        "23784.0 +/- 0.01 million USD",
    )
    feb_2005_row1 = float(
        details.loc["2005-02-01", "nir__row1_outflows"]
    )
    _check(
        checks,
        "nedd_unsigned_outflow_semantics_feb2005",
        abs(feb_2005_row1 - (-3231.52)) < 0.01,
        feb_2005_row1,
        "-3231.52 +/- 0.01 million USD",
    )
    nov_2008_row3 = float(
        details.loc["2008-11-01", "nir__row3_outflows"]
    )
    _check(
        checks,
        "nedd_unsigned_outflow_semantics_nov2008",
        abs(nov_2008_row3 - (-824.0)) < 0.01,
        nov_2008_row3,
        "-824.0 +/- 0.01 million USD",
    )
    may_2011_nir = float(
        details.loc["2011-05-01", "nir__nir_strict_nominal_musd"]
    )
    _check(
        checks,
        "nedd_may2011_imf_gross_outflow_replacement",
        abs(may_2011_nir - 36009.652408) < 0.01,
        may_2011_nir,
        "36009.652408 +/- 0.01 million USD",
    )
    reconciliation_columns = {
        "gross": "nir__pdf_minus_imf_current_gross_musd",
        "row1": "nir__pdf_minus_imf_current_row1_musd",
        "row2": "nir__pdf_minus_imf_current_row2_musd",
        "row3": "nir__pdf_minus_imf_current_row3_musd",
    }
    observed_material_differences: set[str] = set()
    for component, column in reconciliation_columns.items():
        material = details.loc[details[column].abs() > 0.02, column].dropna()
        observed_material_differences.update(
            f"{date:%Y-%m}:{component}"
            for date in material.index
            if date != pd.Timestamp("2011-05-01")
        )
    expected_material_differences = {
        "2000-10:gross",
        "2000-11:gross",
        "2005-12:gross",
        "2007-12:gross",
        "2018-05:row1",
        "2009-02:row3",
        "2011-06:row3",
        "2011-07:row3",
        "2014-09:row3",
        "2019-06:row3",
    }
    _check(
        checks,
        "nedd_archived_pdf_current_imf_difference_allowlist",
        observed_material_differences == expected_material_differences,
        sorted(observed_material_differences),
        sorted(expected_material_differences),
    )
    nir_assumption_count = int(
        details["nir__nir_blank_or_absent_assumption_flag"]
        .astype("string")
        .str.lower()
        .eq("true")
        .sum()
    )
    nir_conservative_count = int(
        details["nir__nir_strict_real_jul2024_musd_na_conservative"]
        .notna()
        .sum()
    )
    _check(
        checks,
        "nedd_blank_assumption_and_conservative_variant",
        nir_assumption_count == 264 and nir_conservative_count == 32,
        {
            "assumption_months": nir_assumption_count,
            "conservative_nonmissing_months": nir_conservative_count,
        },
        {"assumption_months": 264, "conservative_nonmissing_months": 32},
    )
    may_2011_quality = {
        component: str(
            details.loc["2011-05-01", f"nir__{component}_reporting_quality"]
        )
        for component in ("row1", "row2", "row3")
    }
    _check(
        checks,
        "nedd_may2011_missing_components_explicitly_flagged",
        (
            all(value == "absent_assumed_zero" for value in may_2011_quality.values())
            and str(
                details.loc[
                    "2011-05-01", "nir__nir_blank_or_absent_assumption_flag"
                ]
            ).lower()
            == "true"
        ),
        may_2011_quality,
        "row1/row2/row3 absent assumptions flagged; overall conservative flag true",
    )

    raw_imf = (
        PROJECT_ROOT / "data" / "raw" / "snapshots" / selected / "imf_irfcl"
    )
    required_imf_codes = {
        "IRFCLDT1_IRFCL65_USD",
        "IRFCLDT2_IRFCL80_FO_USD",
        "IRFCLDT2_IRFCL79_FO_USD",
        "IRFCLDT2_IRFCL1T_SHP_USD",
        "IRFCLDT2_IRFCL48T_FO_USD",
        "IRFCLDT2_IRFCL50T_FO_USD",
        "IRFCLDT2_IRFCL46T_FO_USD",
        "IRFCLDT2_IRFCL49T_IN_USD",
        "IRFCLDT2_IRFCL50T_IN_USD",
        "IRFCLDT2_IRFCL47T_IN_USD",
        "IRFCLDT2_IRFCL78_USD",
        "IRFCLDT2_IRFCL85_USD",
    }
    codelist_path = raw_imf / "CL_IRFCL_INDICATOR_PUB_4.0.0.xml"
    codelist_root = ET.parse(codelist_path).getroot()
    codelist_codes = {
        element.attrib["id"]
        for element in codelist_root.iter()
        if element.tag.rsplit("}", 1)[-1] == "Code" and "id" in element.attrib
    }

    def imf_observations(code: str) -> dict[pd.Timestamp, float]:
        root = ET.parse(raw_imf / f"{code}.xml").getroot()
        result: dict[pd.Timestamp, float] = {}
        for element in root.iter():
            if (
                element.tag.rsplit("}", 1)[-1] == "Series"
                and element.attrib.get("INDICATOR") == code
            ):
                scale = int(element.attrib.get("SCALE", "0"))
                for observation in element:
                    if observation.tag.rsplit("}", 1)[-1] != "Obs":
                        continue
                    period = observation.attrib["TIME_PERIOD"].replace("-M", "-")
                    result[pd.Timestamp(period + "-01")] = (
                        float(observation.attrib["OBS_VALUE"]) / (10**scale)
                    )
        return result

    imf_observations_by_code = {
        code: imf_observations(code) for code in required_imf_codes
    }
    optional_empty_codes = {
        "IRFCLDT2_IRFCL46T_FO_USD",
        "IRFCLDT2_IRFCL50T_IN_USD",
        "IRFCLDT2_IRFCL47T_IN_USD",
    }
    reverse_repo_expected = {
        pd.Timestamp("2011-05-01"): -0.77,
        pd.Timestamp("2011-06-01"): -5.24,
        pd.Timestamp("2011-07-01"): -2.85,
    }
    reverse_repo_actual = imf_observations_by_code[
        "IRFCLDT2_IRFCL49T_IN_USD"
    ]
    _check(
        checks,
        "imf_irfcl_codelist_empty_series_and_inflow_diagnostics",
        (
            required_imf_codes.issubset(codelist_codes)
            and all(
                (raw_imf / f"{code}.xml").is_file() for code in required_imf_codes
            )
            and all(
                len(imf_observations_by_code[code]) == 0
                for code in optional_empty_codes
            )
            and reverse_repo_actual == reverse_repo_expected
        ),
        {
            "missing_codelist_codes": sorted(
                required_imf_codes.difference(codelist_codes)
            ),
            "optional_empty_observation_counts": {
                code: len(imf_observations_by_code[code])
                for code in sorted(optional_empty_codes)
            },
            "reverse_repo_values": {
                date.strftime("%Y-%m"): value
                for date, value in reverse_repo_actual.items()
            },
        },
        {
            "missing_codelist_codes": [],
            "optional_empty_observation_counts": {
                code: 0 for code in sorted(optional_empty_codes)
            },
            "reverse_repo_values": {
                "2011-05": -0.77,
                "2011-06": -5.24,
                "2011-07": -2.85,
            },
        },
    )

    transition_flag = (
        details["parallel_fx__parallel_fx_source_transition_flag"]
        .astype("string")
        .str.lower()
        .eq("true")
    )
    observed_transition_months = set(
        details.index[transition_flag].strftime("%Y-%m")
    )
    expected_transition_months = {
        "1971-09",
        "1976-01",
        "1981-01",
        "1981-06",
        "1983-01",
        "1990-06",
        "2011-10",
        "2013-01",
        "2016-01",
        "2019-09",
    }
    parallel_baseline = details["parallel_fx__parallel_fx_growth_pct"]
    parallel_same_source = details[
        "parallel_fx__parallel_fx_growth_same_source_pct"
    ]
    non_transition = ~transition_flag
    same_source_difference = (
        parallel_baseline[non_transition] - parallel_same_source[non_transition]
    ).abs().dropna()
    _check(
        checks,
        "parallel_fx_carrier_transitions_and_same_source_sensitivity",
        (
            observed_transition_months == expected_transition_months
            and parallel_same_source[transition_flag].isna().all()
            and parallel_baseline[non_transition]
            .isna()
            .equals(parallel_same_source[non_transition].isna())
            and bool((same_source_difference < 1e-12).all())
        ),
        {
            "carrier_transition_months": sorted(observed_transition_months),
            "same_source_max_difference_elsewhere": (
                float(same_source_difference.max())
                if len(same_source_difference)
                else np.nan
            ),
        },
        {
            "carrier_transition_months": sorted(expected_transition_months),
            "same_source_max_difference_elsewhere": "<1e-12",
        },
    )
    expected_unified_dates = (
        pd.date_range("1967-03-01", "1971-08-01", freq="MS")
        .append(pd.date_range("1981-01-01", "1981-05-01", freq="MS"))
        .append(pd.date_range("1990-06-01", "2011-09-01", freq="MS"))
        .append(pd.date_range("2016-01-01", "2019-08-01", freq="MS"))
    )
    unified_flag = (
        details["parallel_fx__fx_gap_imposed_zero_unified_flag"]
        .astype("string")
        .str.lower()
        .eq("true")
    )
    observed_unified_dates = details.index[unified_flag]
    unified_gaps = details.loc[unified_flag, "parallel_fx__fx_gap_pct"]
    _check(
        checks,
        "parallel_fx_unified_market_gap_zero",
        (
            observed_unified_dates.equals(expected_unified_dates)
            and bool((unified_gaps.abs() < 1e-12).all())
        ),
        {
            "flag_count": int(unified_flag.sum()),
            "max_abs_gap": float(unified_gaps.abs().max()),
        },
        {
            "flag_count": len(expected_unified_dates),
            "max_abs_gap": "<1e-12",
        },
    )
    _check(
        checks,
        "raw_source_hash_map_complete",
        (
            set(source_map["variable_id"]) == set(VARIABLE_COLUMNS)
            and source_map["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
            and source_map["source_role"].ne("").all()
            and not source_map.duplicated(
                ["variable_id", "request_id", "local_path"]
            ).any()
            and {
                "indec_195.1_NIVEL_GENERAL_0_0_13",
                "indec_196.1_NIVEL_GENERAL_2014_0_13",
                "indec_197.1_NIVEL_GENERAL_2014_0_13",
            }.issubset(
                set(
                    source_map.loc[
                        source_map["variable_id"].eq("U03"), "dataset_id"
                    ]
                )
            )
            and "bcra_v4_160"
            not in set(
                source_map.loc[
                    source_map["variable_id"].eq("U13"), "dataset_id"
                ]
            )
        ),
        {
            "variable_ids": sorted(source_map["variable_id"].unique()),
            "rows": len(source_map),
            "bad_hashes": int(
                (~source_map["sha256"].str.fullmatch(r"[0-9a-f]{64}")).sum()
            ),
            "blank_source_roles": int(source_map["source_role"].eq("").sum()),
        },
        (
            "all 13 variables mapped to unique role-labelled raw requests with "
            "SHA-256 hashes; U03 bridge inputs included; U13 excludes rate ID160"
        ),
    )

    coverage = pd.DataFrame(_coverage_rows(wide))
    coverage_path = output / "coverage_validation.csv"
    coverage.to_csv(coverage_path, index=False)
    checks_frame = pd.DataFrame(checks)
    checks_path = output / "dataset_validation_checks.csv"
    checks_frame.to_csv(checks_path, index=False)

    failed_errors = checks_frame.loc[
        (~checks_frame["passed"]) & checks_frame["severity"].eq("error")
    ]
    result = {
        "snapshot_id": selected,
        "validated_at_utc": utc_now_iso(),
        "status": "passed" if failed_errors.empty else "failed",
        "check_count": int(len(checks_frame)),
        "passed_count": int(checks_frame["passed"].sum()),
        "failed_error_count": int(len(failed_errors)),
        "wide_dataset_sha256": sha256_file(wide_path),
        "long_dataset_sha256": sha256_file(long_path),
        "checks_file": safe_relpath(checks_path),
        "coverage_file": safe_relpath(coverage_path),
    }
    report_path = output / "validation_summary.json"
    write_json(report_path, result)
    write_json(METADATA_ROOT / "latest_validation.json", result)

    # Extend the file manifest after validation. The manifest itself is not
    # self-hashed, avoiding a recursive checksum.
    manifest_path = output / "processed_file_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    validation_files = {
        safe_relpath(coverage_path),
        safe_relpath(checks_path),
        safe_relpath(report_path),
    }
    # Revalidation refreshes the three reports but must not accumulate
    # duplicate manifest rows.
    manifest = manifest.loc[~manifest["file"].isin(validation_files)].copy()
    additions = []
    validation_created_at = utc_now_iso()
    for path, rows, columns in (
        (coverage_path, len(coverage), len(coverage.columns)),
        (checks_path, len(checks_frame), len(checks_frame.columns)),
        (report_path, "", ""),
    ):
        additions.append(
            {
                "snapshot_id": selected,
                "file": safe_relpath(path),
                "rows": rows,
                "columns": columns,
                "sha256": sha256_file(path),
                "created_at_utc": validation_created_at,
            }
        )
    manifest = pd.concat([manifest, pd.DataFrame(additions)], ignore_index=True)
    manifest.to_csv(manifest_path, index=False)
    if not failed_errors.empty:
        raise AssertionError(
            "Dataset validation failed: "
            + json.dumps(failed_errors.to_dict(orient="records"), ensure_ascii=False)
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the latest processed build.")
    parser.add_argument("--snapshot-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_latest_build(args.snapshot_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
