from __future__ import annotations

import argparse
import io
import json
import math
import re
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pipeline.common import (
    METADATA_ROOT,
    PROCESSED_ROOT,
    PROJECT_ROOT,
    TARGET_END,
    TARGET_START,
    ensure_output_dirs,
    latest_snapshot_id,
    log_growth,
    month_index,
    monthly_last,
    monthly_mean,
    parse_number_locale,
    pct_growth,
    safe_relpath,
    sha256_file,
    snapshot_path,
    utc_now_iso,
    write_csv_rows,
    write_json,
)


@dataclass
class BuildArtifacts:
    snapshot_id: str
    wide: pd.DataFrame
    details: pd.DataFrame
    provenance: pd.DataFrame
    validation_metrics: list[dict[str, Any]]
    transformations: list[dict[str, Any]]


VARIABLE_COLUMNS = {
    "U01": "u01_arg_cpi_inflation_mom_pct",
    "U02": "u02_registered_wage_growth_mom_pct",
    "U03": "u03_activity_growth_3mma_pct",
    "U04": "u04_nominal_interest_tem_pct",
    "U05": "u05_monetary_base_growth_mom_pct",
    "U06": "u06_m2_growth_mom_pct",
    "U07": "u07_wheat_price_growth_mom_pct",
    "U08": "u08_brent_price_growth_mom_pct",
    "U09": "u09_us_cpi_inflation_mom_pct",
    "U10": "u10_official_fx_growth_mom_pct",
    "U11": "u11_parallel_fx_growth_mom_pct",
    "U12": "u12_fx_gap_pct",
    "U13": "u13_real_net_reserves_mjul2024_usd",
}


def _read_indec_series(raw: Path, series_id: str) -> pd.Series:
    records: list[tuple[str, float | None]] = []
    files = sorted((raw / "indec").glob(f"{series_id}_page_*.json"))
    if not files:
        raise FileNotFoundError(f"No raw INDEC pages for {series_id}")
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend((str(row[0]), row[1]) for row in payload.get("data", []))
    frame = pd.DataFrame(records, columns=["date", "value"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.drop_duplicates("date", keep="last").sort_values("date")
    return frame.set_index("date")["value"].rename(series_id)


def _read_fred_series(raw: Path, series_id: str) -> pd.Series:
    path = raw / "fred" / f"{series_id}_observations.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("observations", [])
    frame = pd.DataFrame(
        [(row.get("date"), row.get("value")) for row in records],
        columns=["date", "value"],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce")
    return frame.drop_duplicates("date").set_index("date")["value"].sort_index().rename(series_id)


def _read_bcra_v4_series(raw: Path, series_id: int) -> pd.Series:
    values: list[tuple[str, float | None]] = []
    files = sorted((raw / "bcra_v4").glob(f"{series_id}_page_*.json"))
    if not files:
        raise FileNotFoundError(f"No BCRA v4 raw pages for {series_id}")
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            values.extend(
                (str(row.get("fecha")), row.get("valor")) for row in result.get("detalle", [])
            )
    frame = pd.DataFrame(values, columns=["date", "value"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return (
        frame.drop_duplicates("date", keep="last")
        .sort_values("date")
        .set_index("date")["value"]
        .rename(f"bcra_v4_{series_id}")
    )


def _read_bcra_panser(raw: Path) -> pd.DataFrame:
    path = raw / "bcra" / "panser.txt"
    frame = pd.read_csv(
        path,
        sep=";",
        names=["series_id", "date", "value"],
        dtype={"series_id": "Int64", "date": str, "value": str},
        encoding="latin-1",
    )
    frame["date"] = pd.to_datetime(frame["date"], format="%d/%m/%Y", errors="coerce")
    frame["value"] = pd.to_numeric(
        frame["value"].str.replace(",", ".", regex=False), errors="coerce"
    )
    # Codes 22 and 25 are retained for the audited activity-credit perimeter
    # comparison.  Code 23 + code 25 is the closest public reconstruction of
    # Forte's "crédito total en pesos"; code 22 is total credit but mixes
    # domestic and foreign currency.
    frame = frame[frame["series_id"].isin([15, 22, 23, 25, 30, 3543])]
    return frame.dropna(subset=["date"]).sort_values(["series_id", "date"])


def _panser_series(frame: pd.DataFrame, series_id: int) -> pd.Series:
    selected = frame.loc[frame["series_id"].eq(series_id), ["date", "value"]]
    selected = selected.copy()
    selected["date"] = selected["date"].dt.to_period("M").dt.to_timestamp()
    return selected.drop_duplicates("date", keep="last").set_index("date")["value"].sort_index()


def _read_bcra_cpi_check(raw: Path) -> pd.Series:
    path = raw / "bcra" / "ES_INFO_SERIES_TASHIS.TXT"
    frame = pd.read_csv(
        path,
        sep=";",
        names=["series_id", "date", "value"],
        dtype={"series_id": "Int64", "date": str, "value": str},
        encoding="latin-1",
    )
    frame = frame[frame["series_id"].eq(7931)]
    frame["date"] = pd.to_datetime(frame["date"], format="%d/%m/%Y", errors="coerce")
    frame["value"] = pd.to_numeric(
        frame["value"].str.replace(",", ".", regex=False), errors="coerce"
    )
    frame["date"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    return frame.drop_duplicates("date").set_index("date")["value"].sort_index()


def _read_cifra(raw: Path) -> pd.Series:
    path = raw / "cifra" / "IPC-Provincias-2007-2018.xlsx"
    table = pd.read_excel(path, sheet_name="Hoja1", header=None)
    # Frozen workbook contract: Excel row 4 onward, A=Período,
    # B=Índice enero 2014=100. Hoja2/Hoja3 are empty.
    dates = pd.to_datetime(table.iloc[3:, 0], errors="coerce")
    values = pd.to_numeric(table.iloc[3:, 1], errors="coerce")
    mask = dates.notna() & values.notna()
    result = pd.Series(values[mask].to_numpy(), index=dates[mask], dtype=float)
    result.index = result.index.to_period("M").to_timestamp()
    result = result[~result.index.duplicated(keep="last")].sort_index().rename("cifra_ipc")
    if (
        len(result) != 144
        or result.index.min() != pd.Timestamp("2007-01-01")
        or result.index.max() != pd.Timestamp("2018-12-01")
        or not result.index.equals(pd.date_range("2007-01-01", "2018-12-01", freq="MS"))
        or not (result > 0).all()
        or not math.isclose(float(result.loc["2007-01-01"]), 22.031247512474046)
        or not math.isclose(float(result.loc["2018-12-01"]), 427.74677040401707)
    ):
        raise ValueError(
            "CIFRA workbook no longer matches the audited 144-month Hoja1 contract."
        )
    return result


def construct_cpi(raw: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    historical = _read_indec_series(raw, "178.1_NL_GENERAL_0_0_13")
    national = _read_indec_series(raw, "148.3_INIVELNAL_DICI_M_26")
    mendoza = _read_indec_series(raw, "195.1_NIVEL_GENERAL_0_0_13")
    neuquen = _read_indec_series(raw, "196.1_NIVEL_GENERAL_2014_0_13")
    san_luis = _read_indec_series(raw, "197.1_NIVEL_GENERAL_2014_0_13")
    cifra = _read_cifra(raw)

    full_idx = pd.date_range("1943-01-01", TARGET_END, freq="MS")
    level = pd.Series(index=full_idx, dtype=float, name="arg_cpi_synthetic_level")
    source = pd.Series(index=full_idx, dtype="string", name="arg_cpi_source")
    pre = historical.loc[: "2006-12-01"].dropna()
    level.loc[pre.index] = pre
    source.loc[pre.index] = "INDEC historical linked IPC-GBA 178.1"

    bridge_inputs: dict[str, float] = {}
    for name, series in (("Mendoza", mendoza), ("Neuquen", neuquen), ("San_Luis", san_luis)):
        pair = series.reindex(pd.to_datetime(["2006-12-01", "2007-01-01"]))
        if pair.notna().all() and (pair > 0).all():
            bridge_inputs[name] = float(100.0 * (pair.iloc[1] / pair.iloc[0] - 1.0))
    if len(bridge_inputs) != 3:
        raise ValueError(f"January-2007 CPI bridge lacks one of three provinces: {bridge_inputs}")
    bridge_pct = float(np.mean(list(bridge_inputs.values())))
    level.loc["2007-01-01"] = level.loc["2006-12-01"] * (1.0 + bridge_pct / 100.0)
    source.loc["2007-01-01"] = "arithmetic mean Mendoza/Neuquen/San Luis simple-rate bridge"

    cifra_growth = pct_growth(cifra)
    for date in pd.date_range("2007-02-01", "2016-12-01", freq="MS"):
        growth = cifra_growth.get(date)
        if pd.isna(growth):
            raise ValueError(f"Missing CIFRA growth at {date:%Y-%m}")
        prior = date - pd.offsets.MonthBegin(1)
        level.loc[date] = level.loc[prior] * (1.0 + float(growth) / 100.0)
        source.loc[date] = "CIFRA IPC Provincias retrospective reconstruction"

    national_growth = pct_growth(national)
    for date in pd.date_range("2017-01-01", TARGET_END, freq="MS"):
        growth = national_growth.get(date)
        if pd.isna(growth):
            raise ValueError(f"Missing national CPI growth at {date:%Y-%m}")
        prior = date - pd.offsets.MonthBegin(1)
        level.loc[date] = level.loc[prior] * (1.0 + float(growth) / 100.0)
        source.loc[date] = "INDEC national CPI level 148.3"

    panel_ns = pd.concat(
        {"neuquen": pct_growth(neuquen), "san_luis": pct_growth(san_luis)}, axis=1
    ).mean(axis=1, skipna=False)
    result = pd.DataFrame(
        {
            "arg_cpi_level": level,
            "arg_cpi_inflation_pct": pct_growth(level),
            "arg_cpi_inflation_log_pct": log_growth(level),
            "arg_cpi_source": source,
            "arg_cpi_panel_neuquen_san_luis_pct": panel_ns,
            "arg_cpi_bcra_7931_simple_mom_check_pct": _read_bcra_cpi_check(raw),
        }
    )
    compare = pd.concat(
        {
            "cifra": cifra_growth.loc["2007-02-01":"2015-12-01"],
            "panel": panel_ns.loc["2007-02-01":"2015-12-01"],
        },
        axis=1,
    ).dropna()
    metrics = {
        "jan2007_bridge_simple_pct": bridge_pct,
        "jan2007_bridge_components": bridge_inputs,
        "panel_vs_cifra_n": int(len(compare)),
        "panel_vs_cifra_corr": float(compare.corr().iloc[0, 1]),
        "panel_vs_cifra_rmse_pp": float(
            np.sqrt(np.mean((compare["panel"] - compare["cifra"]) ** 2))
        ),
        "mendoza_gap_months_2012_04_to_2016_03": int(
            mendoza.reindex(pd.date_range("2012-04-01", "2016-03-01", freq="MS")).isna().sum()
        ),
    }
    return result, metrics


def _run_pdftotext(path: Path, layout: bool = True) -> str:
    command = ["pdftotext"]
    if layout:
        command.append("-layout")
    command.extend([str(path), "-"])
    completed = subprocess.run(command, check=True, capture_output=True)
    return completed.stdout.decode("utf-8", errors="replace").replace("\r", "")


MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "setiembre": 9,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def _letters(value: str) -> str:
    normalized = "".join(
        char
        for char in unicodedata.normalize("NFD", value.lower())
        if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^a-z]", "", normalized)


def _ocr_month(line: str) -> int | None:
    value = _letters(line[:28])
    for label, number in MONTHS.items():
        if label in value or value.startswith(label[:3]):
            return number
    if value.startswith("abn"):
        return 4
    if value.startswith("ocr"):
        return 10
    if value.startswith("d") and ("embre" in value or "cembre" in value):
        return 12
    if value.startswith("n") and "embre" in value:
        return 11
    return None


WAGE_OCR_OVERRIDES = {
    (1967, 5): 0.0000023944,
    (1978, 6): 0.00045203,
    (1978, 7): 0.00045203,
    (1978, 9): 0.00045203,
    (1978, 10): 0.00045203,
    (1981, 2): 0.0042896,
    (1984, 1): 0.27748,
    (1986, 1): 6.862,
    (1986, 10): 14.793,
    (1986, 11): 14.793,
    (1986, 12): 14.793,
}


def _parse_old_wage_token(token: str, year: int, month: int) -> tuple[float, str]:
    if (year, month) in WAGE_OCR_OVERRIDES:
        return WAGE_OCR_OVERRIDES[(year, month)], "audited_override"
    raw = token.strip().lower()
    value = (
        raw.replace("o", "0")
        .replace("b", "6")
        .replace("?", "2")
        .replace("j", "1")
        .replace(" ", "")
        .replace("\xa0", "")
    )
    value = re.sub(r"[^0-9,.\-]", "", value)
    if "," in value and "." in value:
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
    elif "," in value:
        value = value.replace(",", ".")
    if value.count(".") > 1:
        pieces = value.split(".")
        value = "".join(pieces[:-1]) + "." + pieces[-1]
    if "." not in value:
        if value.startswith("0") and len(value) > 1:
            value = "0." + value[1:]
        elif year == 1986 and len(value) >= 4:
            value = value[:-3] + "." + value[-3:]
    return float(value), "automatic_ocr_normalization"


def _parse_historical_isbic(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    text = _run_pdftotext(path, layout=True)
    candidates = [
        match.start()
        for match in re.finditer(r"Personal\s+No\s+Calific", text, flags=re.IGNORECASE)
        if match.start() > len(text) * 0.55
    ]
    if not candidates:
        raise ValueError("Could not locate no-calificado tables in historical wage PDF.")
    lines = text[min(candidates) :].splitlines()
    records: list[dict[str, Any]] = []
    published_means: dict[int, float] = {}
    for index, line in enumerate(lines):
        year_matches = list(re.finditer(r"(?<!\d)(19\d{2})(?!\d)", line))
        if len(year_matches) < 3:
            continue
        years = [int(match.group()) for match in year_matches[:3]]
        if years[0] < 1945 or years[0] > 1987:
            continue
        positions = [match.start() for match in year_matches[:3]]
        boundaries = [
            0,
            (positions[0] + positions[1]) // 2,
            (positions[1] + positions[2]) // 2,
            max(len(line) + 20, positions[2] + 25),
        ]
        for row in lines[index + 1 : index + 15]:
            month = _ocr_month(row)
            is_mean = "prom" in _letters(row[:28])
            if month is None and not is_mean:
                continue
            row_boundaries = boundaries[:-1] + [max(len(row), boundaries[-1])]
            for column, year in enumerate(years):
                left = 20 if column == 0 else row_boundaries[column]
                right = row_boundaries[column + 1]
                token = row[left:right]
                try:
                    value, method = _parse_old_wage_token(token, year, month or 13)
                except (ValueError, TypeError):
                    value, method = float("nan"), "parse_failure"
                if is_mean:
                    if pd.notna(value):
                        published_means[year] = value
                else:
                    records.append(
                        {
                            "date": pd.Timestamp(year=year, month=int(month), day=1),
                            "value": value,
                            "raw_token": token.strip(),
                            "parse_method": method,
                        }
                    )
    audit = (
        pd.DataFrame(records)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .set_index("date")
    )
    target = audit.loc["1965-01-01":"1987-12-01"].copy()
    expected = pd.date_range("1965-01-01", "1987-12-01", freq="MS")
    target = target.reindex(expected)
    if target["value"].isna().any():
        missing = target.index[target["value"].isna()].strftime("%Y-%m").tolist()
        raise ValueError(f"Historical wage transcription has missing months: {missing}")
    checks: list[dict[str, Any]] = []
    for year in range(1965, 1988):
        calculated = float(target.loc[str(year), "value"].mean())
        published = published_means.get(year, float("nan"))
        checks.append(
            {
                "year": year,
                "calculated_mean": calculated,
                "published_ocr_mean": published,
                "relative_difference": (
                    calculated / published - 1 if pd.notna(published) and published != 0 else np.nan
                ),
            }
        )
    check_frame = pd.DataFrame(checks)
    monthly_audit = target.reset_index(names="date")
    monthly_audit.insert(0, "record_type", "monthly_transcription")
    monthly_audit.insert(
        0, "row_key", monthly_audit["date"].dt.strftime("monthly_%Y_%m")
    )
    annual_audit = check_frame.copy()
    annual_audit["date"] = pd.to_datetime(annual_audit["year"].astype(str) + "-12-01")
    annual_audit.insert(0, "record_type", "annual_mean_check")
    annual_audit.insert(0, "row_key", "annual_" + annual_audit["year"].astype(str))
    audit = pd.concat([monthly_audit, annual_audit], ignore_index=True, sort=False)
    if audit["row_key"].duplicated().any():
        raise AssertionError("Historical ISBIC audit row keys are not unique.")
    return target["value"].rename("isbic_no_calificado_old"), audit


def _parse_modern_isbic(path: Path) -> pd.Series:
    text = _run_pdftotext(path, layout=True)
    current_year: int | None = None
    rows: list[tuple[pd.Timestamp, float]] = []
    for line in text.splitlines():
        year_match = re.match(r"^\s*((?:19|20)\d{2})\s*$", line)
        if year_match:
            current_year = int(year_match.group(1))
            continue
        if current_year is None:
            continue
        month = None
        normalized = _letters(line[:20])
        for label, number in MONTHS.items():
            if normalized.startswith(label[:5]):
                month = number
                break
        if month is None:
            continue
        numeric = list(re.finditer(r"[-+]?\d[\d.,]*", line))
        candidates = [match for match in numeric if 78 <= match.start() <= 102]
        if not candidates:
            continue
        token = candidates[0].group()
        # This official Spanish table uses comma as the decimal separator even
        # when exactly three digits follow it (for example 40,336 = 40.336).
        # The generic locale parser intentionally cannot infer that convention.
        cleaned = token.strip().replace(" ", "")
        if "," in cleaned:
            value = float(cleaned.replace(".", "").replace(",", "."))
        else:
            value = float(cleaned)
        rows.append((pd.Timestamp(current_year, month, 1), value))
    result = (
        pd.DataFrame(rows, columns=["date", "value"])
        .drop_duplicates("date", keep="last")
        .set_index("date")["value"]
        .sort_index()
        .rename("isbic_no_calificado_modern")
    )
    expected = pd.date_range("1988-01-01", "1994-07-01", freq="MS")
    missing = expected.difference(result.index)
    if len(missing):
        raise ValueError(f"Modern ISBIC parser missing: {missing.strftime('%Y-%m').tolist()}")
    anchors = {
        pd.Timestamp("1988-01-01"): 40.336,
        pd.Timestamp("1988-12-01"): 184.600,
        pd.Timestamp("1989-01-01"): 198.3,
        pd.Timestamp("1994-07-01"): 219773.1,
    }
    for date, expected_value in anchors.items():
        if not math.isclose(float(result.loc[date]), expected_value, rel_tol=0, abs_tol=1e-6):
            raise ValueError(
                f"Modern ISBIC parser anchor {date:%Y-%m}={result.loc[date]}, "
                f"expected {expected_value}."
            )
    return result


def construct_wages(raw: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    old, old_audit = _parse_historical_isbic(raw / "wages" / "4si9_4.pdf")
    modern = _parse_modern_isbic(raw / "wages" / "indice_isbic.pdf")
    if not math.isclose(float(old.loc["1987-12-01"]), 36.860, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Historical ISBIC Dec-1987 anchor does not equal 36.860.")
    isbic = pd.concat([old.loc[:"1987-12-01"], modern.loc["1988-01-01":"1994-07-01"]])
    isbic = isbic[~isbic.index.duplicated(keep="last")].sort_index()
    ripte = _read_indec_series(raw, "158.1_REPTE_0_0_5")
    link_date = pd.Timestamp("1994-07-01")
    if link_date not in ripte.index or link_date not in isbic.index:
        raise ValueError("The July-1994 ISBIC/RIPTE overlap is absent.")
    ratio = float(isbic.loc[link_date] / ripte.loc[link_date])
    ripte_linked = ripte * ratio
    wage = pd.concat(
        [
            isbic.loc[:link_date],
            ripte_linked.loc[link_date + pd.offsets.MonthBegin(1) : TARGET_END],
        ]
    ).rename("registered_wage_spliced_level")
    source = pd.Series(index=wage.index, dtype="string", name="registered_wage_source")
    source.loc[: "1987-12-01"] = "INDEC 1991 linked no-calificado table"
    source.loc["1988-01-01":"1994-07-01"] = "official ISBIC no-calificado table"
    source.loc["1994-08-01":] = "RIPTE linked at 1994-07"
    result = pd.DataFrame(
        {
            "registered_wage_level": wage,
            "registered_wage_growth_pct": pct_growth(wage),
            "registered_wage_growth_log_pct": log_growth(wage),
            "registered_wage_source": source,
            "isbic_no_calificado_level": isbic,
            "ripte_raw_ars": ripte,
        }
    )
    metrics = {
        "old_isbic_start": old.index.min().date().isoformat(),
        "old_isbic_end": old.index.max().date().isoformat(),
        "modern_isbic_start": modern.index.min().date().isoformat(),
        "ripte_start": ripte.dropna().index.min().date().isoformat(),
        "splice_date": link_date.date().isoformat(),
        "ripte_level_multiplier": ratio,
        "isbic_aug1994_growth_pct": float(
            100.0 * (modern.loc["1994-08-01"] / modern.loc["1994-07-01"] - 1.0)
        ),
        "ripte_aug1994_growth_pct": float(
            100.0 * (ripte.loc["1994-08-01"] / ripte.loc["1994-07-01"] - 1.0)
        ),
        "assumption": "personal no calificado; ISBIC through 1994-07, RIPTE growth from 1994-08",
    }
    return result, metrics, old_audit


def _linked_emae(raw: Path, seasonally_adjusted: bool = False) -> tuple[pd.Series, float]:
    if seasonally_adjusted:
        old_id = "10.3_ISD_1993_M_31"
        new_id = "143.3_NO_PR_2004_A_31"
    else:
        old_id = "10.3_ISOM_1993_M_29"
        new_id = "143.3_NO_PR_2004_A_21"
    old = _read_indec_series(raw, old_id)
    new = _read_indec_series(raw, new_id)
    overlap = pd.concat({"old": old, "new": new}, axis=1).dropna()
    overlap = overlap.loc["2004-01-01":"2013-12-01"]
    if len(overlap) < 100:
        raise ValueError(f"EMAE overlap is unexpectedly short for {old_id}/{new_id}.")
    ratio = float(np.exp(np.median(np.log(overlap["old"] / overlap["new"]))))
    linked = old.loc[:"2003-12-01"].copy()
    new_scaled = new * ratio
    linked = pd.concat([linked, new_scaled.loc["2004-01-01":]])
    linked = linked[~linked.index.duplicated(keep="last")].sort_index()
    linked.name = "emae_linked_sa" if seasonally_adjusted else "emae_linked_original"
    return linked, ratio


def _archival_activity_patches() -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Return audited manual transcriptions from two immutable official PDFs.

    The source scans have no usable text layer. Values are deliberately kept
    here (rather than interpolated) so every inserted observation is reviewable
    and traceable to a printed table and page.
    """

    auto_1991 = [
        6302,
        2998,
        6121,
        10686,
        11586,
        9794,
        13294,
        15060,
        15018,
        15655,
        16793,
        15651,
    ]
    auto_1992 = [
        17688,
        5293,
        18108,
        20701,
        19880,
        23152,
        24678,
        25173,
        25902,
        26017,
        26829,
        28601,
    ]
    steel_1991 = [
        279.7,
        218.1,
        223.8,
        251.2,
        245.3,
        272.8,
        281.3,
        255.6,
        292.8,
        260.2,
        245.2,
        146.0,
    ]
    steel_1992 = [
        100.5,
        180.4,
        246.4,
        245.8,
        239.0,
        286.8,
        250.8,
        211.9,
        216.3,
        240.6,
        239.4,
        222.0,
    ]
    dates_1991 = pd.date_range("1991-01-01", "1991-12-01", freq="MS")
    dates_1992 = pd.date_range("1992-01-01", "1992-12-01", freq="MS")
    autos = pd.Series(
        auto_1991 + auto_1992,
        index=dates_1991.append(dates_1992),
        dtype=float,
        name="archival_total_vehicle_units",
    )
    steel = pd.Series(
        steel_1991 + steel_1992,
        index=dates_1991.append(dates_1992),
        dtype=float,
        name="archival_crude_steel_thousand_tonnes",
    )
    if float(autos.loc["1991"].sum()) != 138958 or float(autos.loc["1992"].sum()) != 262022:
        raise AssertionError("Archival automobile transcription does not match printed annual totals.")
    if not math.isclose(float(steel.loc["1992"].sum()), 2679.9, abs_tol=1e-9):
        raise AssertionError("Archival 1992 steel transcription does not match the printed total.")

    audit = pd.DataFrame(
        {
            "date": autos.index,
            "total_vehicle_units": autos.to_numpy(),
            "crude_steel_thousand_tonnes": steel.to_numpy(),
            "vehicle_source_file": np.where(
                autos.index.year == 1991,
                "activity/informe_economico_1992.pdf, PDF p120 / table AII-5",
                (
                    "activity/estadisticas_productos_industriales_1993_12.pdf, "
                    "PDF p31"
                ),
            ),
            "steel_source_file": np.where(
                steel.index.year == 1991,
                "activity/informe_economico_1992.pdf, PDF p119 / table AII-4",
                (
                    "activity/estadisticas_productos_industriales_1993_12.pdf, "
                    "PDF p23"
                ),
            ),
            "transcription_method": "manual double-check of printed table; no interpolation",
        }
    )
    return autos, steel, audit


def _fit_activity_log_level_proxy(
    components: pd.DataFrame,
    emae: pd.Series,
    *,
    fit_start: str,
    fit_end: str,
    mapping_start: str = "1993-01-01",
    mapping_end: str = "2013-12-01",
) -> dict[str, Any]:
    train = components.loc[fit_start:fit_end].dropna()
    expected = pd.date_range(fit_start, fit_end, freq="MS")
    missing = expected.difference(train.index)
    if len(train) < 240:
        raise ValueError(
            f"Only {len(train)} complete PCA months for {fit_start}:{fit_end}; "
            f"missing examples={missing[:12].strftime('%Y-%m').tolist()}"
        )
    center = train.mean()
    scale = train.std(ddof=0)
    if (scale <= 0).any():
        raise ValueError("A PCA component has zero training variance.")
    standardized = (components - center) / scale
    pca = PCA(n_components=1).fit(standardized.loc[train.index])
    complete = standardized.dropna()
    scores = pd.Series(
        pca.transform(complete)[:, 0],
        index=complete.index,
        name="activity_pc1_log_levels",
    )
    overlap = pd.concat(
        {"score": scores, "log_emae": np.log(emae.where(emae > 0))}, axis=1
    ).dropna()
    overlap = overlap.loc[mapping_start:mapping_end]
    if len(overlap) < 200:
        raise ValueError("The PCA/EMAE mapping overlap is unexpectedly short.")
    if overlap["score"].corr(overlap["log_emae"]) < 0:
        scores = -scores
        pca.components_ = -pca.components_
        overlap["score"] = -overlap["score"]
    slope, intercept = np.polyfit(
        overlap["score"].to_numpy(), overlap["log_emae"].to_numpy(), 1
    )
    predicted = np.exp(intercept + slope * scores).rename("activity_pca_predicted_level")
    anchor = float(emae.loc["1993-01-01"] / predicted.loc["1993-01-01"])
    predicted *= anchor
    validation = pd.concat(
        {
            "predicted_level": predicted,
            "emae_level": emae,
            "predicted_growth": log_growth(predicted),
            "emae_growth": log_growth(emae),
        },
        axis=1,
    ).loc[mapping_start:mapping_end]
    return {
        "predicted": predicted,
        "scores": scores,
        "center": center,
        "scale": scale,
        "loadings": pca.components_[0],
        "explained_variance": float(pca.explained_variance_ratio_[0]),
        "train": train,
        "missing": missing,
        "slope": float(slope),
        "intercept": float(intercept),
        "anchor": anchor,
        "log_level_corr": float(
            np.log(validation[["predicted_level", "emae_level"]]).corr().iloc[0, 1]
        ),
        "growth_corr": float(
            validation[["predicted_growth", "emae_growth"]].dropna().corr().iloc[0, 1]
        ),
    }


def construct_activity(
    raw: Path, panser: pd.DataFrame, cpi: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    credit_private_nominal = _panser_series(panser, 23)
    credit_public_nominal = _panser_series(panser, 25)
    credit_nominal = (
        credit_private_nominal + credit_public_nominal
    ).rename("total_private_and_public_nonfinancial_peso_credit")
    cpi_level = cpi["arg_cpi_level"].reindex(credit_nominal.index)
    credit_real = (credit_nominal / cpi_level).rename("credit_real_cpi_deflated")
    credit_private_real = (
        credit_private_nominal
        / cpi["arg_cpi_level"].reindex(credit_private_nominal.index)
    ).rename("credit_private_only_real_cpi_deflated")
    # "Resto" equals all non-car commercial categories in the historical
    # source. Cars + the narrower API "utilitarios" column omits a material
    # category and is therefore not the preferred total.
    autos_cars_api = _read_indec_series(raw, "330.3_PRODUCCIONLES__22")
    autos_resto_api = _read_indec_series(raw, "330.3_PRODUCCIONSTO__16")
    autos_api = (autos_cars_api + autos_resto_api).rename("total_vehicle_units")
    steel_api = _read_indec_series(raw, "359.3_ACERO_CRUDUDO__11").rename(
        "raw_steel_thousand_tonnes"
    )
    # The API observations are checked against the separately archived
    # official distribution files.  This catches mnemonic drift and unit
    # changes before any archival patch is applied.
    auto_file = pd.read_csv(
        raw
        / "activity"
        / "datos-historicos-industria-automotriz-unidades-mensuales.csv"
    )
    auto_file["indice_tiempo"] = pd.to_datetime(auto_file["indice_tiempo"])
    auto_file = auto_file.set_index("indice_tiempo").sort_index()
    autos_official_file = (
        pd.to_numeric(auto_file["produccion_automoviles"], errors="coerce")
        + pd.to_numeric(auto_file["produccion_resto"], errors="coerce")
    ).rename("total_vehicle_units_official_file")
    steel_file = pd.read_csv(
        raw
        / "activity"
        / "datos-historicos-industria-siderurgica-datos-mensuales.csv"
    )
    steel_file["indice_tiempo"] = pd.to_datetime(steel_file["indice_tiempo"])
    steel_official_file = (
        steel_file.set_index("indice_tiempo")["acero_crudo"]
        .pipe(pd.to_numeric, errors="coerce")
        .sort_index()
        .rename("steel_official_file")
    )

    def assert_api_file_equal(
        api: pd.Series, official_file: pd.Series, label: str
    ) -> None:
        comparison = pd.concat(
            {
                "api": api.loc[:TARGET_END],
                "official_file": official_file.loc[:TARGET_END],
            },
            axis=1,
        )
        if not comparison["api"].isna().equals(comparison["official_file"].isna()):
            mismatch = comparison.index[
                comparison["api"].isna() != comparison["official_file"].isna()
            ]
            raise ValueError(
                f"{label} API/file missingness differs at "
                f"{mismatch[:12].strftime('%Y-%m').tolist()}"
            )
        values = comparison.dropna()
        if not np.allclose(
            values["api"].to_numpy(),
            values["official_file"].to_numpy(),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(f"{label} API values differ from official CSV.")

    assert_api_file_equal(autos_api, autos_official_file, "automobile total")
    assert_api_file_equal(steel_api, steel_official_file, "crude steel")
    archival_autos, archival_steel, transcription_audit = _archival_activity_patches()
    if not autos_api.reindex(archival_autos.index).isna().all():
        raise ValueError(
            "The official automobile API now contains 1991-1992 values; "
            "stop rather than overwrite a future backfill with the archival transcription."
        )
    if not steel_api.reindex(archival_steel.index).isna().all():
        raise ValueError(
            "The official steel API now contains 1991-1992 values; stop rather "
            "than overwrite a future backfill with the archival transcription."
        )
    autos = autos_api.copy()
    autos.loc[archival_autos.index] = archival_autos
    steel = steel_api.copy()
    steel.loc[archival_steel.index] = archival_steel
    emae, emae_ratio = _linked_emae(raw, seasonally_adjusted=False)
    emae_sa, emae_sa_ratio = _linked_emae(raw, seasonally_adjusted=True)

    components = pd.concat(
        {
            "log_real_credit": np.log(credit_real.where(credit_real > 0)),
            "log_auto_production": np.log(autos.where(autos > 0)),
            "log_steel_production": np.log(steel.where(steel > 0)),
        },
        axis=1,
    ).sort_index()
    private_credit_components = components.copy()
    private_credit_components["log_real_credit"] = np.log(
        credit_private_real.where(credit_private_real > 0)
    )
    # The preferred Forte-like retrospective variant estimates the PC over the
    # clean EMAE overlap. It reproduces the reported >80% level correlation but
    # is not a real-time construction. A pre-1989 sensitivity freezes loadings
    # before the public automobile gap.
    primary = _fit_activity_log_level_proxy(
        components, emae, fit_start="1993-01-01", fit_end="2013-12-01"
    )
    private_credit_sensitivity = _fit_activity_log_level_proxy(
        private_credit_components,
        emae,
        fit_start="1993-01-01",
        fit_end="2013-12-01",
    )
    pre1989 = _fit_activity_log_level_proxy(
        components, emae, fit_start="1965-01-01", fit_end="1988-12-01"
    )
    predicted_level = primary["predicted"]
    scores = primary["scores"]
    activity_level = predicted_level.loc[:"1992-12-01"].copy()
    activity_level = pd.concat([activity_level, emae.loc["1993-01-01":]])
    activity_level = activity_level[~activity_level.index.duplicated(keep="last")].sort_index()
    activity_level = activity_level.reindex(
        pd.date_range(activity_level.index.min(), activity_level.index.max(), freq="MS")
    )
    activity_level.name = "activity_spliced_level"

    sensitivity_level = pre1989["predicted"].loc[:"1992-12-01"].copy()
    sensitivity_level = pd.concat([sensitivity_level, emae.loc["1993-01-01":]])
    sensitivity_level = sensitivity_level[~sensitivity_level.index.duplicated(keep="last")]
    sensitivity_level = sensitivity_level.reindex(activity_level.index)

    # Growth-PCA is retained only as a diagnostic rejected alternative.
    growth_components = components.diff()
    growth_train = growth_components.loc["1993-02-01":"2013-12-01"].dropna()
    growth_center = growth_train.mean()
    growth_scale = growth_train.std(ddof=0)
    growth_z = (growth_components - growth_center) / growth_scale
    growth_pca = PCA(n_components=1).fit(growth_z.loc[growth_train.index])
    growth_complete = growth_z.dropna()
    growth_score = pd.Series(
        growth_pca.transform(growth_complete)[:, 0],
        index=growth_complete.index,
        name="activity_pc1_log_changes",
    )
    emae_growth = np.log(emae.where(emae > 0)).diff()
    growth_overlap = pd.concat({"score": growth_score, "emae": emae_growth}, axis=1).dropna()
    growth_overlap = growth_overlap.loc["1993-02-01":"2013-12-01"]
    if growth_overlap["score"].corr(growth_overlap["emae"]) < 0:
        growth_overlap["score"] *= -1
        growth_score *= -1

    growth_pca_corr = float(growth_overlap.corr().iloc[0, 1])

    activity_growth = pct_growth(activity_level, "activity_growth_pct")
    activity_growth_log = log_growth(activity_level, "activity_growth_log_pct")
    sensitivity_growth = pct_growth(
        sensitivity_level, "activity_growth_pre1989_pca_pct"
    )
    source = pd.Series(index=activity_level.index, dtype="string", name="activity_source")
    source.loc[:"1992-12-01"] = (
        "retrospective PC1 log levels: real private+public nonfinancial peso "
        "credit/total vehicles/crude steel"
    )
    source.loc["1989-01-01":"1990-12-01"] = (
        "missing: no defensible public monthly total-vehicle observations"
    )
    source.loc["1993-01-01":] = "linked EMAE original NSA"
    result = pd.DataFrame(
        {
            "activity_level": activity_level,
            "activity_growth_pct": activity_growth,
            "activity_growth_log_pct": activity_growth_log,
            "activity_growth_3mma_pct": activity_growth.rolling(3, min_periods=3).mean(),
            "activity_source": source,
            "activity_level_pre1989_pca_sensitivity": sensitivity_level,
            "activity_growth_3mma_pre1989_pca_sensitivity_pct": sensitivity_growth.rolling(
                3, min_periods=3
            ).mean(),
            "activity_pca_predicted_level": predicted_level,
            "activity_pc1_score": scores,
            "emae_linked_original": emae,
            "emae_linked_sa": emae_sa,
            "real_credit_level": credit_real,
            "real_credit_private_only_sensitivity_level": credit_private_real,
            "auto_production_units": autos,
            "auto_production_api_before_archival_patch_units": autos_api,
            "auto_production_official_file_units": autos_official_file,
            "steel_production_thousand_tonnes": steel,
            "steel_production_api_before_archival_patch_thousand_tonnes": steel_api,
            "steel_production_official_file_thousand_tonnes": steel_official_file,
        }
    )
    loadings = pd.DataFrame(
        {
            "component": components.columns,
            "mean_primary_1993_2013": primary["center"].values,
            "std_primary_1993_2013": primary["scale"].values,
            "pc1_loading_primary": primary["loadings"],
            "mean_sensitivity_1965_1988": pre1989["center"].values,
            "std_sensitivity_1965_1988": pre1989["scale"].values,
            "pc1_loading_sensitivity": pre1989["loadings"],
            "alignment_slope_sensitivity_1965_1988": pre1989["slope"],
            "alignment_intercept_sensitivity_1965_1988": pre1989["intercept"],
            "jan1993_anchor_sensitivity_1965_1988": pre1989["anchor"],
            "explained_variance_sensitivity_1965_1988": pre1989[
                "explained_variance"
            ],
            "pc1_loading_private_credit_only_sensitivity": (
                private_credit_sensitivity["loadings"]
            ),
            "alignment_slope_private_credit_only_sensitivity": (
                private_credit_sensitivity["slope"]
            ),
            "alignment_intercept_private_credit_only_sensitivity": (
                private_credit_sensitivity["intercept"]
            ),
            "jan1993_anchor_private_credit_only_sensitivity": (
                private_credit_sensitivity["anchor"]
            ),
            "explained_variance_private_credit_only_sensitivity": (
                private_credit_sensitivity["explained_variance"]
            ),
        }
    )
    holdout = pd.concat(
        {
            "predicted_level": predicted_level,
            "emae_level": emae,
            "predicted_growth": log_growth(predicted_level),
            "emae_growth": log_growth(emae),
        },
        axis=1,
    ).loc["2014-01-01":"2015-12-01"]
    holdout_level = np.log(
        holdout[["predicted_level", "emae_level"]]
    ).dropna()
    holdout_growth = holdout[
        ["predicted_growth", "emae_growth"]
    ].dropna()
    missing_pre1993 = activity_level.reindex(
        pd.date_range("1965-01-01", "1992-12-01", freq="MS")
    )
    metrics = {
        "pca_training_start": primary["train"].index.min().date().isoformat(),
        "pca_training_end": primary["train"].index.max().date().isoformat(),
        "pca_training_n": int(len(primary["train"])),
        "pca_missing_training_months": primary["missing"].strftime("%Y-%m").tolist(),
        "pc1_explained_variance_ratio": primary["explained_variance"],
        "level_proxy_log_level_corr_with_emae_1993_2013": primary["log_level_corr"],
        "level_proxy_growth_corr_with_emae_growth_1993_2013": primary["growth_corr"],
        "postfit_holdout_log_level_corr_2014_2015": float(
            holdout_level.corr().iloc[0, 1]
        ),
        "postfit_holdout_growth_corr_2014_2015": float(
            holdout_growth.corr().iloc[0, 1]
        ),
        "pre1989_loading_variant_log_level_corr_with_emae_1993_2013": pre1989[
            "log_level_corr"
        ],
        "pre1989_loading_variant_alignment_slope": pre1989["slope"],
        "pre1989_loading_variant_alignment_intercept": pre1989["intercept"],
        "pre1989_loading_variant_jan1993_anchor": pre1989["anchor"],
        "pre1989_loading_variant_explained_variance_ratio": pre1989[
            "explained_variance"
        ],
        "private_credit_only_sensitivity_log_level_corr_with_emae_1993_2013": (
            private_credit_sensitivity["log_level_corr"]
        ),
        "private_credit_only_sensitivity_growth_corr_with_emae_1993_2013": (
            private_credit_sensitivity["growth_corr"]
        ),
        "growth_pca_corr_with_emae_growth_1993_2013": growth_pca_corr,
        "alignment_slope": primary["slope"],
        "alignment_intercept": primary["intercept"],
        "jan1993_anchor_multiplier": primary["anchor"],
        "emae_base2004_to_1993_ratio_original": emae_ratio,
        "emae_base2004_to_1993_ratio_sa": emae_sa_ratio,
        "archival_auto_1991_sum": float(archival_autos.loc["1991"].sum()),
        "archival_auto_1992_sum": float(archival_autos.loc["1992"].sum()),
        "archival_steel_1991_sum": float(archival_steel.loc["1991"].sum()),
        "archival_steel_1992_sum": float(archival_steel.loc["1992"].sum()),
        "archival_steel_1991_printed_annual_total": 2972.3,
        "archival_steel_1991_monthly_sum_minus_printed_total": float(
            archival_steel.loc["1991"].sum() - 2972.3
        ),
        "missing_pre1993_activity_months": missing_pre1993.index[
            missing_pre1993.isna()
        ].strftime("%Y-%m").tolist(),
        "chosen_variant": (
            "PC1 of standardized log levels using CPI-deflated BCRA codes 23+25 "
            "(private and public nonfinancial peso credit), loadings and log-level "
            "mapping estimated 1993-2013; Jan-1993 anchored; original NSA EMAE. "
            "Retrospective and not vintage-safe. Code-23-only and pre-1989-loading "
            "sensitivities are retained."
        ),
        "known_gap": (
            "1989-01 through 1990-12 remains NA because no complete official public "
            "monthly total-vehicle table was found."
        ),
    }
    return result, metrics, loadings, transcription_audit


def _read_pink_sheet(raw: Path) -> pd.DataFrame:
    path = raw / "world_bank" / "CMO-Historical-Data-Monthly.xlsx"
    frame = pd.read_excel(path, sheet_name="Monthly Prices", header=4)
    date_column = frame.columns[0]
    dates = pd.to_datetime(
        frame[date_column].astype(str).str.replace("M", "-", regex=False) + "-01",
        format="%Y-%m-%d",
        errors="coerce",
    )
    result = pd.DataFrame(index=dates)
    for column in ("Wheat, US HRW", "Crude oil, Brent", "Crude oil, average"):
        if column not in frame.columns:
            raise KeyError(f"Pink Sheet column is absent: {column}")
        result[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy()
    return result.loc[result.index.notna()].sort_index()


def construct_standard_variables(
    raw: Path, panser: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = _panser_series(panser, 15).rename("monetary_base_level")
    m2 = _panser_series(panser, 3543).rename("m2_total_level")
    deposit_tna = _panser_series(panser, 30).rename("deposit_30_59d_tna_pct")
    policy_daily = _read_bcra_v4_series(raw, 160)
    policy_tna = monthly_mean(policy_daily).rename("policy_tna_pct")
    interest_tna = deposit_tna.copy().rename("nominal_interest_tna_pct")
    interest_tna.loc["2015-12-01":] = policy_tna.loc["2015-12-01":]
    interest_source = pd.Series(index=interest_tna.index, dtype="string")
    interest_source.loc[:"2015-11-01"] = "BCRA code 30 deposits 30-59 days"
    interest_source.loc["2015-12-01":] = "BCRA v4 ID 160 policy rate"
    interest_tem = (interest_tna * 30.0 / 365.0).rename("nominal_interest_tem_pct")
    interest_tem_compound = (
        100.0 * ((1.0 + interest_tna / 100.0) ** (30.0 / 365.0) - 1.0)
    ).rename("nominal_interest_tem_compound_sensitivity_pct")

    pink = _read_pink_sheet(raw)
    wheat = pink["Wheat, US HRW"].rename("wheat_us_hrw_usd_per_mt")
    brent = pink["Crude oil, Brent"].rename("brent_usd_per_barrel")
    crude_average = pink["Crude oil, average"].rename("crude_average_usd_per_barrel")
    us_cpi = _read_fred_series(raw, "CPIAUCNS").rename("us_cpi_level")
    official_fx = _read_fred_series(raw, "ARGCCUSMA02STM").rename(
        "official_fx_ars_per_usd"
    )
    result = pd.DataFrame(
        {
            "nominal_interest_tna_pct": interest_tna,
            "nominal_interest_tem_pct": interest_tem,
            "nominal_interest_tem_compound_sensitivity_pct": interest_tem_compound,
            "nominal_interest_source": interest_source,
            "monetary_base_level": base,
            "monetary_base_growth_pct": pct_growth(base),
            "monetary_base_growth_log_pct": log_growth(base),
            "m2_total_level": m2,
            "m2_growth_pct": pct_growth(m2),
            "m2_growth_log_pct": log_growth(m2),
            "wheat_us_hrw_usd_per_mt": wheat,
            "wheat_growth_pct": pct_growth(wheat),
            "wheat_growth_log_pct": log_growth(wheat),
            "brent_usd_per_barrel": brent,
            "brent_growth_pct": pct_growth(brent),
            "brent_growth_log_pct": log_growth(brent),
            "crude_average_usd_per_barrel": crude_average,
            "us_cpi_level": us_cpi,
            "us_cpi_inflation_pct": pct_growth(us_cpi),
            "us_cpi_inflation_log_pct": log_growth(us_cpi),
            "official_fx_ars_per_usd": official_fx,
            "official_fx_growth_pct": pct_growth(official_fx),
            "official_fx_growth_log_pct": log_growth(official_fx),
        }
    )
    brent_eia = monthly_mean(_read_fred_series(raw, "DCOILBRENTEU"))
    overlap = pd.concat({"pink": brent, "eia": brent_eia}, axis=1).dropna()
    overlap = overlap.loc["1987-05-01":TARGET_END]
    wheat_fred = _read_fred_series(raw, "PWHEAMTUSDM")
    wheat_overlap = pd.concat({"pink": wheat, "fred": wheat_fred}, axis=1).dropna()
    metrics = {
        "interest_formula": "TEM percent = TNA percent * 30 / 365 (simple nominal convention)",
        "interest_switch": "2015-12",
        "brent_pink_vs_eia_level_corr": float(overlap.corr().iloc[0, 1]),
        "brent_pink_vs_eia_growth_corr": float(
            pd.concat(
                {"pink": log_growth(overlap["pink"]), "eia": log_growth(overlap["eia"])},
                axis=1,
            )
            .dropna()
            .corr()
            .iloc[0, 1]
        ),
        "wheat_pink_vs_fred_level_corr": float(wheat_overlap.corr().iloc[0, 1]),
        "brent_pre_1987_status": (
            "World Bank column is labelled Brent back to 1960 but predates a comparable "
            "Brent spot market; treated as a weak historical backcast/proxy."
        ),
    }
    return result, metrics


def _normalize_historical_fx(
    series: pd.Series, kind: str
) -> tuple[pd.Series, pd.Series]:
    values = series.copy().astype(float)
    normalized = pd.Series(index=values.index, dtype=float)
    factor = pd.Series(index=values.index, dtype=float)
    if kind == "financial":
        mask = values.notna()
        normalized.loc[mask] = values.loc[mask] / 1e11
        factor.loc[mask] = 1e11
    elif kind == "free":
        spans = [
            ("1976-01-08", "1981-01-01", 1e11),
            ("1983-01-03", "1983-06-01", 1e8),
            ("1983-06-02", "1985-06-18", 1e7),
            ("1985-06-19", "1990-01-03", 1e4),
            ("1990-01-04", "1992-01-01", 1e2),
        ]
        for start, end, divisor in spans:
            mask = values.index.to_series().between(pd.Timestamp(start), pd.Timestamp(end))
            normalized.loc[mask] = values.loc[mask] / divisor
            factor.loc[mask] = divisor
    else:
        raise ValueError(f"Unknown historical FX type: {kind}")
    return normalized.rename(f"{kind}_fx_normalized"), factor.rename(f"{kind}_fx_divisor")


def _weekday_monthly_mean(series: pd.Series) -> pd.Series:
    selected = series.loc[series.index.dayofweek < 5]
    return monthly_mean(selected)


def _read_ambito(raw: Path, kind: str) -> pd.Series:
    rows: list[tuple[str, Any]] = []
    for path in sorted((raw / "ambito").glob(f"{kind}_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload or not isinstance(payload, list):
            raise ValueError(f"Empty or invalid Ámbito payload: {path}")
        header = [str(value).strip().lower() for value in payload[0]]
        if kind == "blue":
            if len(header) < 3 or "fecha" not in header[0] or "venta" not in header[2]:
                raise ValueError(f"Unexpected Ámbito blue header {payload[0]} in {path}")
        elif len(header) < 2 or "fecha" not in header[0] or not any(
            token in header[1] for token in ("referencia", "valor", "ccl")
        ):
            raise ValueError(f"Unexpected Ámbito CCL header {payload[0]} in {path}")
        file_rows = 0
        for row in payload[1:]:
            if kind == "blue":
                if len(row) >= 3:
                    rows.append((row[0], row[2]))
                    file_rows += 1
            elif len(row) >= 2:
                rows.append((row[0], row[1]))
                file_rows += 1
        if file_rows == 0:
            raise ValueError(f"Ámbito yearly payload has no observations: {path}")
    frame = pd.DataFrame(rows, columns=["date", "value"])
    frame["date"] = pd.to_datetime(frame["date"], format="%d/%m/%Y", errors="coerce")
    def parse_quote(value: Any) -> float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        token = str(value).strip().replace(" ", "")
        if "," in token:
            return float(token.replace(".", "").replace(",", "."))
        return float(token)

    frame["value"] = frame["value"].map(parse_quote)
    frame = frame.dropna().drop_duplicates("date", keep="last").sort_values("date")
    if frame.empty or frame["date"].duplicated().any() or not (frame["value"] > 0).all():
        raise ValueError(f"Ámbito parsed history is invalid for {kind}.")
    return frame.set_index("date")["value"].rename(f"ambito_{kind}")


def _monthly_quote_quality(series: pd.Series, prefix: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, group in series.dropna().sort_index().groupby(series.dropna().index.to_period("M")):
        month_start = period.to_timestamp()
        month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
        expected = len(pd.bdate_range(month_start, month_end))
        first = group.index.min()
        last = group.index.max()
        coverage = len(group) / expected if expected else np.nan
        rows.append(
            {
                "date": month_start,
                f"{prefix}_quote_count": int(len(group)),
                f"{prefix}_first_quote_date": first.date().isoformat(),
                f"{prefix}_last_quote_date": last.date().isoformat(),
                f"{prefix}_business_day_coverage_ratio": coverage,
                f"{prefix}_partial_month_flag": bool(
                    coverage < 0.5
                    or first >= month_start + pd.Timedelta(days=7)
                    or last < month_end - pd.Timedelta(days=7)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def construct_parallel_fx(
    raw: Path, official_fx: pd.Series
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    financial_raw = _read_indec_series(raw, "175.1_DR_FINANTA_0_0_22")
    free_raw = _read_indec_series(raw, "175.1_DR_LIBRNTA_0_0_17")
    financial_norm, financial_factor = _normalize_historical_fx(financial_raw, "financial")
    free_norm, free_factor = _normalize_historical_fx(free_raw, "free")
    financial_monthly = _weekday_monthly_mean(financial_norm)
    free_monthly = _weekday_monthly_mean(free_norm)
    blue_daily = _read_ambito(raw, "blue")
    ccl_daily = _read_ambito(raw, "ccl")
    blue_monthly = monthly_mean(blue_daily)
    ccl_monthly = monthly_mean(ccl_daily)
    a3500_monthly = pd.Series(dtype=float, name="a3500_monthly_control")
    if list((raw / "indec").glob("175.1_DR_REFE500_0_0_25_page_*.json")):
        a3500_monthly = _weekday_monthly_mean(
            _read_indec_series(raw, "175.1_DR_REFE500_0_0_25")
        ).rename("a3500_monthly_control")
    financial_quality = _monthly_quote_quality(
        financial_norm.dropna().loc[lambda x: x.index.dayofweek < 5], "financial"
    )
    free_quality = _monthly_quote_quality(
        free_norm.dropna().loc[lambda x: x.index.dayofweek < 5], "free"
    )
    blue_quality = _monthly_quote_quality(blue_daily, "blue")
    ccl_quality = _monthly_quote_quality(ccl_daily, "ccl")

    index = pd.date_range("1965-01-01", TARGET_END, freq="MS")
    parallel = pd.Series(index=index, dtype=float, name="parallel_fx_ars_per_usd")
    source = pd.Series(index=index, dtype="string", name="parallel_fx_source")
    regime = pd.Series(index=index, dtype="string", name="parallel_fx_regime")

    def assign(start: str, end: str, values: pd.Series, label: str, regime_label: str) -> None:
        target = pd.date_range(start, end, freq="MS")
        parallel.loc[target] = values.reindex(target)
        source.loc[target] = label
        regime.loc[target] = regime_label

    # No public monthly series is silently invented for the early-control or
    # controlled-period source gaps.
    assign(
        "1967-03-01",
        "1971-08-01",
        official_fx,
        "OECD/FRED official monthly average",
        "documented_unified_market",
    )
    assign(
        "1971-09-01",
        "1975-12-01",
        financial_monthly,
        "BCRA financial sale, weekday mean, normalized",
        "controlled_financial_market",
    )
    assign(
        "1976-01-01",
        "1981-01-01",
        free_monthly,
        "BCRA free sale, weekday mean, normalized",
        "free_market",
    )
    assign(
        "1981-01-01",
        "1981-05-01",
        official_fx,
        "OECD/FRED official monthly average",
        "documented_unified_market",
    )
    assign(
        "1981-06-01",
        "1981-12-01",
        financial_monthly,
        "BCRA financial sale, weekday mean, normalized",
        "controlled_financial_market",
    )
    assign(
        "1982-07-01",
        "1982-12-01",
        financial_monthly,
        "BCRA financial sale, weekday mean, normalized",
        "controlled_financial_market",
    )
    assign(
        "1983-01-01",
        "1987-01-01",
        free_monthly,
        "BCRA free sale, weekday mean, normalized",
        "free_market",
    )
    assign(
        "1987-11-01",
        "1989-04-01",
        free_monthly,
        "BCRA free sale, weekday mean, normalized",
        "free_market",
    )
    assign(
        "1990-02-01",
        "1990-05-01",
        free_monthly,
        "BCRA free sale, weekday mean, normalized",
        "controlled_market_undocumented_scale_factor",
    )
    assign(
        "1990-06-01",
        "2011-09-01",
        official_fx,
        "OECD/FRED official monthly average",
        "documented_unified_market",
    )
    assign(
        "2011-10-01",
        "2012-12-01",
        blue_monthly,
        "Ambito informal sale quote",
        "informal_market_proxy",
    )
    assign(
        "2013-01-01",
        "2015-12-01",
        ccl_monthly,
        "Ambito/Rava CCL reference",
        "ccl_proxy",
    )
    assign(
        "2016-01-01",
        "2019-08-01",
        official_fx,
        "OECD/FRED official monthly average",
        "documented_unified_market",
    )
    assign(
        "2019-09-01",
        "2024-07-01",
        ccl_monthly,
        "Ambito/Rava CCL reference",
        "ccl_proxy",
    )

    gap = 100.0 * (parallel / official_fx - 1.0)
    growth = pct_growth(parallel)
    previous_observed_source = source.where(parallel.notna()).ffill().shift()
    source_transition = (
        parallel.notna()
        & previous_observed_source.notna()
        & source.ne(previous_observed_source)
    )
    block_start = parallel.notna() & (
        ~parallel.notna().shift(1, fill_value=False) | source_transition
    )
    result = pd.DataFrame(
        {
            "parallel_fx_level": parallel,
            "parallel_fx_growth_pct": growth,
            "parallel_fx_growth_same_source_pct": growth.mask(source_transition),
            "parallel_fx_growth_log_pct": log_growth(parallel),
            "parallel_fx_source": source,
            "parallel_fx_regime": regime,
            "parallel_fx_previous_observed_source": previous_observed_source,
            "parallel_fx_source_transition_flag": source_transition,
            "parallel_fx_observed_block_start_flag": block_start,
            "fx_gap_imposed_zero_unified_flag": (
                regime.eq("documented_unified_market") & parallel.notna()
            ),
            "fx_gap_pct": gap,
            "ambito_blue_level": blue_monthly,
            "ambito_ccl_level": ccl_monthly,
            "bcra_financial_normalized": financial_monthly,
            "bcra_free_normalized": free_monthly,
            "official_fx_level": official_fx,
            "a3500_monthly_control": a3500_monthly,
        }
    )
    for quality in (financial_quality, free_quality, blue_quality, ccl_quality):
        result = result.join(quality)
    controlled_missing = pd.date_range("1982-01-01", "1982-06-01", freq="MS").append(
        pd.date_range("1987-02-01", "1987-10-01", freq="MS")
    ).append(
        pd.date_range("1989-05-01", "1990-01-01", freq="MS")
    )
    partial_months: dict[str, list[str]] = {}
    for label, quality in (
        ("financial", financial_quality),
        ("free", free_quality),
        ("blue", blue_quality),
        ("ccl", ccl_quality),
    ):
        flag = quality.get(f"{label}_partial_month_flag", pd.Series(dtype=bool))
        partial_months[label] = (
            quality.index[flag.fillna(False)].strftime("%Y-%m").tolist()
            if len(quality)
            else []
        )
    metrics = {
        "first_nonmissing": parallel.dropna().index.min().date().isoformat(),
        "uninterrupted_public_start": "1990-06 (official unified); Ambito observed from 2002-01-11",
        "controlled_gap_month_count": int(parallel.reindex(controlled_missing).isna().sum()),
        "pre_1971_missing_month_count": int(
            parallel.reindex(pd.date_range("1965-01-01", "1971-08-01", freq="MS")).isna().sum()
        ),
        "ambito_blue_first_date": blue_daily.index.min().date().isoformat(),
        "ambito_ccl_first_date": ccl_daily.index.min().date().isoformat(),
        "monthly_aggregation": (
            "BCRA: Monday-Friday calendar rows only (holidays cannot be identified); "
            "Ambito: mean of returned quote dates; official: source monthly average"
        ),
        "free_1990_factor_status": (
            "factor 100 inferred from the API's undocumented 1990-01-04 rescale and "
            "official-rate level check; selected only Feb-May 1990 and low-confidence"
        ),
        "modern_regime_rule": (
            "official through 2011-09; blue 2011-10..2012-12; CCL 2013-01..2015-12; "
            "official 2016-01..2019-08; CCL from 2019-09"
        ),
        "source_transition_months": (
            source_transition.index[source_transition]
            .strftime("%Y-%m")
            .tolist()
        ),
        "observed_block_start_months": (
            block_start.index[block_start].strftime("%Y-%m").tolist()
        ),
        "source_transition_baseline_growth_pct": {
            date.strftime("%Y-%m"): float(growth.loc[date])
            for date in source_transition.index[source_transition]
            if pd.notna(growth.loc[date])
        },
        "same_source_growth_sensitivity": (
            "parallel_fx_growth_same_source_pct is NA whenever the selected "
            "source changes; baseline U11 retains the economically observed "
            "regime-boundary change and exposes the flag."
        ),
        "unified_gap_rule": (
            "U11 level equals U10 in documented unified regimes, so U12 is "
            "imposed to zero rather than measured from a parallel quote."
        ),
        "partial_months_by_source": partial_months,
        "financial_nonnull_blocks": _nonnull_blocks(financial_raw),
        "free_nonnull_blocks": _nonnull_blocks(free_raw),
    }
    if len(a3500_monthly):
        official_check = pd.concat(
            {"fred_oecd": official_fx, "a3500": a3500_monthly}, axis=1
        ).dropna()
        official_check = official_check.loc["2002-03-01":TARGET_END]
        metrics["official_fred_vs_a3500_level_corr"] = float(
            official_check.corr().iloc[0, 1]
        )
        metrics["official_fred_vs_a3500_mean_abs_pct_difference"] = float(
            (100.0 * (official_check["fred_oecd"] / official_check["a3500"] - 1.0))
            .abs()
            .mean()
        )
    normalization_audit = pd.concat(
        {
            "financial_raw": financial_raw,
            "financial_normalized": financial_norm,
            "financial_divisor": financial_factor,
            "free_raw": free_raw,
            "free_normalized": free_norm,
            "free_divisor": free_factor,
        },
        axis=1,
    )
    return result, metrics, normalization_audit


def _nonnull_blocks(series: pd.Series) -> list[dict[str, str]]:
    dates = series.dropna().index.sort_values()
    if len(dates) == 0:
        return []
    blocks: list[dict[str, str]] = []
    start = dates[0]
    previous = dates[0]
    for date in dates[1:]:
        # The API contains calendar-daily rows. More than seven days is treated
        # as a genuine inactive/missing block rather than a holiday gap.
        if (date - previous).days > 7:
            blocks.append({"start": start.date().isoformat(), "end": previous.date().isoformat()})
            start = date
        previous = date
    blocks.append({"start": start.date().isoformat(), "end": previous.date().isoformat()})
    return blocks


NEDD_NUMBER_RE = re.compile(
    r"(?<![\w])[-+]?(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d+)?)(?![\w])"
)


def _plain(value: str) -> str:
    normalized = "".join(
        char
        for char in unicodedata.normalize("NFD", value.lower())
        if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"\s+", " ", normalized)


def _numbers_in_column_block(
    line: str, x: int, next_total_x: int | None
) -> list[float]:
    values: list[float] = []
    upper = (
        (x + next_total_x) // 2
        if next_total_x is not None
        else len(line) + 1
    )
    for match in NEDD_NUMBER_RE.finditer(line):
        if match.start() < max(0, x - 4) or match.start() >= upper:
            continue
        try:
            values.append(parse_number_locale(match.group()))
        except ValueError:
            continue
    return values


def _first_value_near(
    lines: list[str],
    row_index: int,
    total_x: int,
    next_total_x: int | None,
    offsets: tuple[int, ...] = (0, -1, 1),
) -> float | None:
    for offset in offsets:
        target = row_index + offset
        if target < 0 or target >= len(lines):
            continue
        values = _numbers_in_column_block(lines[target], total_x, next_total_x)
        if values:
            # A superscript footnote can survive at the left edge, but total_x
            # filters it. The first remaining value is the monetary-authority total.
            return float(values[0])
    return None


def _find_line(lines: list[str], *needles: str, start: int = 0) -> int | None:
    normalized_needles = [_plain(needle) for needle in needles]
    for index in range(start, len(lines)):
        value = _plain(lines[index])
        if all(needle in value for needle in normalized_needles):
            return index
    return None


def _nedd_section(lines: list[str]) -> tuple[list[str], int, int | None]:
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if re.search(
                r"\bII\.\s+(?:Flujos|Predetermined)", line, flags=re.IGNORECASE
            )
        ),
        None,
    )
    if start is None:
        raise ValueError("Section II not found")
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if re.search(
                r"\bIII\.\s+(?:Flujos|Contingent)",
                lines[index],
                flags=re.IGNORECASE,
            )
        ),
        len(lines),
    )
    section = lines[start:end]
    total_x = None
    next_total_x = None
    for line in section[:15]:
        matches = list(re.finditer(r"\bTotal\b", line, flags=re.IGNORECASE))
        if matches:
            total_x = matches[0].start()
            next_total_x = matches[1].start() if len(matches) > 1 else None
            break
    if total_x is None:
        raise ValueError("Monetary-authority total column not found")
    return section, total_x, next_total_x


def _parse_nedd_gross(lines: list[str]) -> tuple[float, str]:
    section_two = next(
        (
            index
            for index, line in enumerate(lines)
            if re.search(
                r"\bII\.\s+(?:Flujos|Predetermined)",
                line,
                flags=re.IGNORECASE,
            )
        ),
        len(lines),
    )
    first_page = lines[:section_two]
    for index, line in enumerate(first_page):
        normalized = _plain(line)
        if "a. activos de reserva oficiales" in normalized:
            numbers = [parse_number_locale(match.group()) for match in NEDD_NUMBER_RE.finditer(line)]
            plausible = [value for value in numbers if value > 100]
            if plausible:
                # Monetary authority is the first value; a second value, when
                # present, belongs to central government and must not be used.
                return float(plausible[0]), "section_I_A_monetary_authority"
        if "a. official reserve assets" in normalized:
            numbers = [parse_number_locale(match.group()) for match in NEDD_NUMBER_RE.finditer(line)]
            plausible = [value for value in numbers if value > 100]
            if plausible:
                return float(plausible[0]), "section_I_A_monetary_authority_english"
    for line in first_page:
        normalized = _plain(line)
        if normalized.lstrip().startswith("i. activos de reserva"):
            numbers = [parse_number_locale(match.group()) for match in NEDD_NUMBER_RE.finditer(line)]
            plausible = [value for value in numbers if value > 100]
            if plausible:
                return float(plausible[0]), "section_I_total_legacy_template"
        if normalized.lstrip().startswith("i. official reserve assets"):
            numbers = [parse_number_locale(match.group()) for match in NEDD_NUMBER_RE.finditer(line)]
            plausible = [value for value in numbers if value > 100]
            if plausible:
                return float(plausible[0]), "section_I_total_legacy_template_english"
    raise ValueError("NEDD gross official reserve assets not found")


def _category_aggregate(
    section: list[str], total_x: int, next_total_x: int | None, category: int
) -> tuple[float | None, str]:
    patterns = {
        1: [("1.", "prestamos"), ("1.", "foreign currency loans")],
        2: [
            ("2.", "posiciones"),
            ("2.", "posisiciones"),
            ("2.", "aggregate short and long"),
        ],
        3: [("3.", "otros"), ("3.", "other")],
    }[category]
    index = None
    for pattern in patterns:
        index = _find_line(section, *pattern)
        if index is not None:
            break
    if index is None:
        return None, "category_label_absent"
    # Categories can wrap onto one continuation line before their subrows.
    value = _first_value_near(
        section, index, total_x, next_total_x, offsets=(0, 1, 2, 3)
    )
    return value, "category_aggregate" if value is not None else "category_aggregate_absent"


def _parse_nedd_outflows(
    section: list[str],
    total_x: int,
    next_total_x: int | None,
    reference_date: pd.Timestamp,
) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    methods: dict[str, str] = {}

    # Category 1: gross capital and interest outflows. Legacy templates expose
    # only an aggregate total, which is used as a documented fallback.
    category1_index = _find_line(section, "1.", "prestamos")
    if category1_index is None:
        category1_index = _find_line(section, "1.", "foreign currency loans")
    category2_index = _find_line(section, "2.", "posiciones")
    if category2_index is None:
        category2_index = _find_line(section, "2.", "posisiciones")
    if category2_index is None:
        category2_index = _find_line(section, "2.", "aggregate short and long")
    row1_parts: list[float] = []
    if category1_index is not None:
        stop = category2_index if category2_index is not None else min(
            len(section), category1_index + 10
        )
        for index in range(category1_index + 1, stop):
            normalized = _plain(section[index])
            if "ingresos" in normalized or "inflows" in normalized:
                break
            if not any(
                token in normalized
                for token in ("capital", "intereses", "principal", "interest")
            ):
                continue
            value = _first_value_near(
                section, index, total_x, next_total_x, offsets=(0,)
            )
            label = (
                "capital"
                if "capital" in normalized or "principal" in normalized
                else "interest"
            )
            if value is not None and value < 0:
                row1_parts.append(value)
                methods[f"row1_{label}"] = "monetary_authority_detail_outflow"
            elif (
                value is not None
                and reference_date == pd.Timestamp("2005-02-01")
                and label == "capital"
                and abs(value - 3231.52) < 0.01
            ):
                # The archived February-2005 PDF omits the printed minus sign
                # on this semantically negative capital-outflow row. The IMF
                # country-authority series independently confirms -3231.52.
                row1_parts.append(-abs(value))
                methods[f"row1_{label}"] = (
                    "known_unsigned_outflow_anomaly_normalized"
                )
    if row1_parts:
        values["row1_outflows"] = float(sum(row1_parts))
        methods["row1"] = (
            "sum_negative_capital_interest;"
            "known_unsigned_outflow_anomaly_normalized"
            if methods.get("row1_capital")
            == "known_unsigned_outflow_anomaly_normalized"
            else "sum_negative_capital_interest"
        )
    else:
        aggregate, method = _category_aggregate(
            section, total_x, next_total_x, 1
        )
        values["row1_outflows"] = float(min(aggregate or 0.0, 0.0))
        methods["row1"] = f"{method}_fallback"

    # Category 2: gross short position only. A long position is an inflow and
    # is deliberately not netted in the strict liabilities proxy.
    short_index = _find_line(section, "posiciones cortas")
    if short_index is None:
        short_index = _find_line(section, "short positions")
    short_value = (
        _first_value_near(
            section, short_index, total_x, next_total_x, offsets=(0,)
        )
        if short_index is not None
        else None
    )
    if short_value is None:
        short_value = 0.0
        methods["row2"] = (
            "short_position_absent_assumed_zero"
            if short_index is None
            else "short_position_blank_assumed_zero"
        )
    else:
        methods["row2"] = "gross_short_position"
    values["row2_short"] = float(-abs(short_value) if short_value != 0 else 0.0)

    # Category 3: add only explicitly negative repos, trade credit and payables.
    detail_needles = [
        ("egresos relacionados con recompras", "repos"),
        ("outflows related to repos", "repos"),
        ("credito comercial (-)", "trade_credit"),
        ("trade credit (-)", "trade_credit"),
        ("otras cuentas a pagar", "payables"),
        ("other accounts payable", "payables"),
    ]
    other_parts: list[float] = []
    for needle, label in detail_needles:
        index = _find_line(section, needle)
        if index is None:
            continue
        value = _first_value_near(
            section, index, total_x, next_total_x, offsets=(0,)
        )
        if value is not None and value < 0:
            other_parts.append(value)
            methods[f"row3_{label}"] = "detail_outflow"
        elif (
            value is not None
            and reference_date == pd.Timestamp("2008-11-01")
            and label == "repos"
            and abs(value - 824.0) < 0.01
        ):
            # The archived November-2008 PDF prints the repo outflow as a
            # positive magnitude. Its category total and IMF series establish
            # the omitted-minus anomaly.
            other_parts.append(-abs(value))
            methods[f"row3_{label}"] = (
                "known_unsigned_outflow_anomaly_normalized"
            )
    if other_parts:
        values["row3_outflows"] = float(sum(other_parts))
        methods["row3"] = (
            "sum_negative_detail_outflows;"
            "known_unsigned_outflow_anomaly_normalized"
            if methods.get("row3_repos")
            == "known_unsigned_outflow_anomaly_normalized"
            else "sum_negative_detail_outflows"
        )
    else:
        aggregate, method = _category_aggregate(
            section, total_x, next_total_x, 3
        )
        values["row3_outflows"] = float(min(aggregate or 0.0, 0.0))
        methods["row3"] = f"{method}_fallback"
    return values, methods


def _nedd_embedded_month(lines: list[str]) -> pd.Timestamp | None:
    header = _plain(" ".join(lines[:18]))
    numeric = re.search(
        r"\b\d{1,2}[./-](\d{1,2})[./-](\d{2,4})\b", header
    )
    if numeric:
        month = int(numeric.group(1))
        year_token = int(numeric.group(2))
        year = year_token + (2000 if year_token < 90 else 1900) if year_token < 100 else year_token
        return pd.Timestamp(year, month, 1)
    for label, month in MONTHS.items():
        match = re.search(
            rf"\b{label}\s*[,.-]?\s*(?:de\s+)?(\d{{4}})\b", header
        )
        if match:
            return pd.Timestamp(int(match.group(1)), month, 1)
    # Audited source typo in the September-2005 Spanish template.
    typo = re.search(r"\bsepteimbre\s*[,.-]?\s*(\d{4})\b", header)
    if typo:
        return pd.Timestamp(int(typo.group(1)), 9, 1)
    english_months_full = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    # Legacy English templates use headings such as "as of May 31, 2000".
    # Parse this before abbreviated month-year forms so the day cannot be
    # mistaken for a two-digit year.
    for label, month in english_months_full.items():
        match = re.search(
            rf"\b{label}\s+\d{{1,2}}\s*,?\s*(\d{{4}})\b",
            header,
        )
        if match:
            return pd.Timestamp(int(match.group(1)), month, 1)
    english_months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    for label, month in english_months.items():
        match = re.search(rf"\b{label}[-\s](\d{{4}})\b", header)
        if match:
            return pd.Timestamp(int(match.group(1)), month, 1)
        match = re.search(rf"\b{label}-(\d{{2}})\b", header)
        if match:
            token = int(match.group(1))
            return pd.Timestamp(
                token + (2000 if token < 90 else 1900),
                month,
                1,
            )
    return None


def _parse_nedd_pdf(path: Path, date: pd.Timestamp) -> dict[str, Any]:
    text = _run_pdftotext(path, layout=True)
    lines = text.splitlines()
    embedded_month = _nedd_embedded_month(lines)
    if embedded_month is None:
        raise ValueError("NEDD embedded as-of month not found in PDF header")
    if embedded_month.to_period("M") != date.to_period("M"):
        raise ValueError(
            f"NEDD filename month {date:%Y-%m} disagrees with embedded "
            f"header {embedded_month:%Y-%m}"
        )
    gross, gross_method = _parse_nedd_gross(lines)
    section, total_x, next_total_x = _nedd_section(lines)
    outflows, methods = _parse_nedd_outflows(
        section,
        total_x,
        next_total_x,
        date,
    )
    total_outflows = float(sum(outflows.values()))
    return {
        "date": date,
        "nedd_official_reserve_assets_musd": gross,
        **outflows,
        "gross_predetermined_outflows_musd": total_outflows,
        "nir_strict_nominal_musd": gross + total_outflows,
        "gross_parse_method": gross_method,
        "row1_parse_method": methods.get("row1", ""),
        "row2_parse_method": methods.get("row2", ""),
        "row3_parse_method": methods.get("row3", ""),
        "total_column_x": total_x,
        "next_entity_total_column_x": next_total_x,
        "embedded_as_of_month": embedded_month,
        "filename_embedded_month_match": True,
        "source_file": safe_relpath(path),
    }


def _nedd_last_modified_dates(raw: Path) -> pd.Series:
    manifest_path = raw / "retrieval_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    selected = manifest.loc[manifest["dataset_id"].eq("bcra_nedd_pdf")].copy()
    selected["date"] = pd.to_datetime(
        selected["observation_start_requested"], errors="coerce"
    ).dt.to_period("M").dt.to_timestamp()
    modified = pd.to_datetime(
        selected["source_last_modified"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    return pd.Series(
        modified.to_numpy(), index=selected["date"], name="nedd_source_last_modified"
    )


IMF_IRFCL_VALID_EMPTY_ARG_CODES = {
    "IRFCLDT2_IRFCL46T_FO_USD",
    "IRFCLDT2_IRFCL50T_IN_USD",
    "IRFCLDT2_IRFCL47T_IN_USD",
}


def _read_imf_irfcl(
    raw: Path, indicator: str
) -> tuple[pd.Series, dict[str, Any]]:
    path = raw / "imf_irfcl" / f"{indicator}.xml"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected IMF IRFCL raw response is absent: {path}"
        )
    root = ET.parse(path).getroot()
    series_scale = 0
    series_found = False
    dataset_found = False
    observations: list[tuple[pd.Timestamp, float]] = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "DataSet":
            dataset_found = True
        if tag == "Series" and element.attrib.get("INDICATOR") == indicator:
            series_found = True
            series_scale = int(element.attrib.get("SCALE", "0"))
            for observation in element:
                if observation.tag.rsplit("}", 1)[-1] != "Obs":
                    continue
                period = observation.attrib.get("TIME_PERIOD", "").replace("-M", "-")
                value = observation.attrib.get("OBS_VALUE")
                if not period or value is None:
                    continue
                observations.append(
                    (pd.Timestamp(period + "-01"), float(value) / (10**series_scale))
                )
    if not dataset_found:
        raise ValueError(f"IMF response for {indicator} is not a valid SDMX DataSet.")
    valid_empty = len(observations) == 0 and indicator in IMF_IRFCL_VALID_EMPTY_ARG_CODES
    if not observations and not valid_empty:
        raise ValueError(
            f"IMF response for {indicator} contains no observations and is not "
            "an audited valid-empty Argentina detail series."
        )
    result = pd.Series(
        dict(observations), dtype=float, name=indicator
    ).sort_index()
    status = {
        "indicator": indicator,
        "source_file": safe_relpath(path),
        "series_element_found": series_found,
        "observation_count": int(len(result)),
        "first_observation": (
            result.index.min().date().isoformat() if len(result) else ""
        ),
        "last_observation": (
            result.index.max().date().isoformat() if len(result) else ""
        ),
        "valid_empty_argentina_response": valid_empty,
    }
    return result, status


def construct_nir(raw: Path, us_cpi: pd.Series) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in sorted((raw / "nedd").glob("????-??.pdf")):
        date = pd.Timestamp(path.stem + "-01")
        try:
            rows.append(_parse_nedd_pdf(path, date))
        except Exception as exc:
            failures.append(
                {"date": date, "source_file": safe_relpath(path), "error": repr(exc)}
            )
    valid_pdf_count = len(rows)
    # The live BCRA archive serves June-2011 twice (its May link is wrong).
    # Current official IMF SDMX data recover May-2011 without inventing a
    # monthly observation. The original May-2000 BCRA document is recovered
    # from a fixed Internet Archive capture by the retrieval stage.
    imf_codes = {
        "gross": "IRFCLDT1_IRFCL65_USD",
        "row1_principal": "IRFCLDT2_IRFCL80_FO_USD",
        "row1_interest": "IRFCLDT2_IRFCL79_FO_USD",
        "row1_aggregate": "IRFCLDT2_IRFCL78_USD",
        "row2": "IRFCLDT2_IRFCL1T_SHP_USD",
        "row3_repos": "IRFCLDT2_IRFCL48T_FO_USD",
        "row3_trade_credit": "IRFCLDT2_IRFCL50T_FO_USD",
        "row3_payables": "IRFCLDT2_IRFCL46T_FO_USD",
        "row3_aggregate": "IRFCLDT2_IRFCL85_USD",
        "row3_reverse_repo_inflow": "IRFCLDT2_IRFCL49T_IN_USD",
        "row3_trade_credit_inflow": "IRFCLDT2_IRFCL50T_IN_USD",
        "row3_receivables_inflow": "IRFCLDT2_IRFCL47T_IN_USD",
    }
    raw_manifest = pd.read_csv(raw / "retrieval_manifest.csv", dtype=str).fillna("")
    expected_imf_datasets = {f"imf_irfcl_{code}" for code in imf_codes.values()}
    missing_manifest_datasets = expected_imf_datasets.difference(
        set(raw_manifest["dataset_id"])
    )
    if missing_manifest_datasets:
        raise ValueError(
            "IMF IRFCL files are not fully represented in the raw retrieval "
            f"manifest: {sorted(missing_manifest_datasets)}"
        )
    imf: dict[str, pd.Series] = {}
    imf_status: list[dict[str, Any]] = []
    for name, code in imf_codes.items():
        series, status = _read_imf_irfcl(raw, code)
        imf[name] = series
        imf_status.append({"component_name": name, **status})
    replacement_date = pd.Timestamp("2011-05-01")
    if replacement_date not in {row["date"] for row in rows}:
        required_replacement_observations = (
            "gross",
            "row1_principal",
            "row3_repos",
            "row1_aggregate",
            "row3_aggregate",
        )
        missing_replacement_observations = [
            name
            for name in required_replacement_observations
            if replacement_date not in imf[name].index
        ]
        if missing_replacement_observations:
            raise ValueError(
                "Required IMF detail observations for the May-2011 archive "
                f"repair are absent: {missing_replacement_observations}"
            )
        allowed_absent_at_replacement = {
            "row1_interest",
            "row2",
            "row3_trade_credit",
            "row3_payables",
        }

        def negative_sum(names: list[str]) -> tuple[float, list[str]]:
            values: list[float] = []
            assumptions: list[str] = []
            for name in names:
                if replacement_date in imf[name].index:
                    values.append(min(float(imf[name].loc[replacement_date]), 0.0))
                elif name in allowed_absent_at_replacement:
                    values.append(0.0)
                    code = imf_codes[name]
                    status = next(
                        item for item in imf_status if item["indicator"] == code
                    )
                    qualifier = (
                        "valid-empty-series"
                        if status["valid_empty_argentina_response"]
                        else "month-absent"
                    )
                    assumptions.append(f"{name}:{qualifier}-assumed-zero")
                else:
                    raise ValueError(
                        f"Unapproved missing IMF component {name} at 2011-05."
                    )
            return float(sum(values)), assumptions

        row1_value, row1_assumptions = negative_sum(
            ["row1_principal", "row1_interest"]
        )
        row2_value, row2_assumptions = negative_sum(["row2"])
        row3_value, row3_assumptions = negative_sum(
            ["row3_repos", "row3_trade_credit", "row3_payables"]
        )

        replacement = {
            "date": replacement_date,
            "nedd_official_reserve_assets_musd": float(imf["gross"].loc[replacement_date]),
            "row1_outflows": row1_value,
            "row2_short": row2_value,
            "row3_outflows": row3_value,
            "gross_parse_method": "IMF_SDMX_official_replacement_for_wrong_BCRA_archive_link",
            "row1_parse_method": (
                "IMF_SDMX_principal_observed; interest_absent_assumed_zero"
            ),
            "row2_parse_method": "IMF_SDMX_short_position_absent_assumed_zero",
            "row3_parse_method": (
                "IMF_SDMX_repo_observed; trade_credit_absent_assumed_zero; "
                "payables_valid_empty_series_assumed_zero; "
                "net_category_aggregate_retained_as_diagnostic"
            ),
            "imf_replacement_component_assumptions": ";".join(
                row1_assumptions + row2_assumptions + row3_assumptions
            ),
            "total_column_x": np.nan,
            "next_entity_total_column_x": np.nan,
            "embedded_as_of_month": replacement_date,
            "filename_embedded_month_match": False,
            "source_file": safe_relpath(
                raw / "imf_irfcl" / f"{imf_codes['gross']}.xml"
            ),
        }
        replacement["gross_predetermined_outflows_musd"] = float(
            replacement["row1_outflows"]
            + replacement["row2_short"]
            + replacement["row3_outflows"]
        )
        replacement["nir_strict_nominal_musd"] = float(
            replacement["nedd_official_reserve_assets_musd"]
            + replacement["gross_predetermined_outflows_musd"]
        )
        rows.append(replacement)
        failures = [
            failure for failure in failures if failure["date"] != replacement_date
        ]
    parsed = pd.DataFrame(rows).set_index("date").sort_index()
    failure_frame = pd.DataFrame(failures)
    parsed["nedd_source_last_modified"] = _nedd_last_modified_dates(raw)
    for component in ("row1", "row2", "row3"):
        method_column = f"{component}_parse_method"
        method = parsed[method_column].astype("string")
        parsed[f"{component}_reporting_quality"] = np.select(
            [
                method.str.contains("blank", case=False, na=False),
                method.str.contains("absent", case=False, na=False),
            ],
            ["blank_assumed_zero", "absent_assumed_zero"],
            default="reported_value_or_explicit_zero",
        )
    reporting_quality_columns = [
        "row1_reporting_quality",
        "row2_reporting_quality",
        "row3_reporting_quality",
    ]
    parsed["nir_blank_or_absent_assumption_flag"] = parsed[
        reporting_quality_columns
    ].apply(
        lambda row: any(str(value).endswith("assumed_zero") for value in row),
        axis=1,
    )
    parsed["nir_strict_nominal_musd_na_conservative"] = parsed[
        "nir_strict_nominal_musd"
    ].mask(parsed["nir_blank_or_absent_assumption_flag"])
    imf_row1_details = pd.concat(
        [imf["row1_principal"], imf["row1_interest"]],
        axis=1,
    )
    imf_row1_negative_detail_sum = (
        imf_row1_details.clip(upper=0.0).fillna(0.0).sum(axis=1)
    ).where(imf["gross"].reindex(imf_row1_details.index).notna())
    imf_row3_details = pd.concat(
        [
            imf["row3_repos"],
            imf["row3_trade_credit"],
            imf["row3_payables"],
        ],
        axis=1,
    )
    imf_row3_negative_detail_sum = (
        imf_row3_details.clip(upper=0.0).fillna(0.0).sum(axis=1)
    ).where(imf["gross"].reindex(imf_row3_details.index).notna())
    imf_comparators = {
        "gross": imf["gross"],
        # The strict PDF formula deliberately retains negative detail outflows
        # and ignores positive adjustments. Compare like with like; July-2006
        # is an audited example where the official aggregate includes +0.16.
        "row1": imf_row1_negative_detail_sum,
        "row2": imf["row2"],
        "row3": imf_row3_negative_detail_sum,
    }
    parsed_columns = {
        "gross": "nedd_official_reserve_assets_musd",
        "row1": "row1_outflows",
        "row2": "row2_short",
        "row3": "row3_outflows",
    }
    for component, current_imf in imf_comparators.items():
        aligned = current_imf.reindex(parsed.index)
        if component == "row2":
            # IMF XML omits months with no reported gross short position.
            # Treat that as zero only when an IMF gross observation establishes
            # that the country-month itself is present.
            aligned = aligned.fillna(0.0).where(
                imf["gross"].reindex(parsed.index).notna()
            )
        parsed[f"imf_current_{component}_musd"] = aligned
        parsed[f"pdf_minus_imf_current_{component}_musd"] = (
            parsed[parsed_columns[component]] - aligned
        )
    parsed["imf_current_row1_net_category_aggregate_musd"] = imf[
        "row1_aggregate"
    ].reindex(parsed.index)
    parsed["imf_current_row3_net_category_aggregate_musd"] = imf[
        "row3_aggregate"
    ].reindex(parsed.index)
    parsed["imf_current_reverse_repo_inflow_musd"] = imf[
        "row3_reverse_repo_inflow"
    ].reindex(parsed.index)
    parsed["imf_current_trade_credit_inflow_musd"] = imf[
        "row3_trade_credit_inflow"
    ].reindex(parsed.index)
    parsed["imf_current_receivables_inflow_musd"] = imf[
        "row3_receivables_inflow"
    ].reindex(parsed.index)

    gross_daily = _read_bcra_v4_series(raw, 1)
    gross_eom = monthly_last(gross_daily).rename("bcra_v4_gross_reserves_eom_musd")
    current_accounts = monthly_last(_read_bcra_v4_series(raw, 1243)).rename(
        "bcra_fx_current_accounts_eom_musd"
    )
    passive_repo = monthly_last(_read_bcra_v4_series(raw, 76)).rename(
        "bcra_passive_usd_repo_eom_musd"
    )
    parsed = parsed.join(gross_eom).join(current_accounts).join(passive_repo)
    parsed["predetermined_liability_magnitude_musd"] = -parsed[
        "gross_predetermined_outflows_musd"
    ]
    parsed["nir_bcra_gross_strict_nominal_musd"] = (
        parsed["bcra_v4_gross_reserves_eom_musd"]
        + parsed["gross_predetermined_outflows_musd"]
    )
    parsed["nir_narrow_nominal_musd"] = (
        parsed["bcra_v4_gross_reserves_eom_musd"]
        - parsed["bcra_fx_current_accounts_eom_musd"]
        - parsed["bcra_passive_usd_repo_eom_musd"].clip(lower=0)
    )
    base_cpi = float(us_cpi.loc["2024-07-01"])
    deflator = (base_cpi / us_cpi).rename("usd_jul2024_deflator")
    parsed = parsed.join(deflator)
    parsed["nir_strict_real_jul2024_musd"] = (
        parsed["nir_strict_nominal_musd"] * parsed["usd_jul2024_deflator"]
    )
    parsed["nir_strict_real_jul2024_musd_na_conservative"] = (
        parsed["nir_strict_nominal_musd_na_conservative"]
        * parsed["usd_jul2024_deflator"]
    )
    parsed["nir_bcra_gross_strict_real_jul2024_musd"] = (
        parsed["nir_bcra_gross_strict_nominal_musd"]
        * parsed["usd_jul2024_deflator"]
    )
    parsed["nir_narrow_real_jul2024_musd"] = (
        parsed["nir_narrow_nominal_musd"] * parsed["usd_jul2024_deflator"]
    )

    july = parsed.loc[pd.Timestamp("2024-07-01")]
    material_reconciliation: list[dict[str, Any]] = []
    for component in imf_comparators:
        column = f"pdf_minus_imf_current_{component}_musd"
        selected = parsed.loc[parsed[column].abs() > 0.02, column].dropna()
        for date, value in selected.items():
            if date == replacement_date:
                continue
            key = f"{date:%Y-%m}:{component}"
            if key == "2009-02:row3":
                classification = "detail_vs_net_aggregate_definition"
                interpretation = (
                    "The archived PDF row agrees with the net category aggregate; "
                    "the strict comparator retains only negative gross detail outflows."
                )
            elif key in {"2011-06:row3", "2011-07:row3"}:
                classification = "net_aggregate_includes_reverse_repo_inflow"
                reverse_repo = imf["row3_reverse_repo_inflow"].get(date, np.nan)
                interpretation = (
                    "The IMF net category aggregate incorporates the separately "
                    f"reported reverse-repo inflow ({reverse_repo} million USD); "
                    "the strict gross-outflow formula excludes inflows."
                )
            elif key in {
                "2000-10:gross",
                "2000-11:gross",
                "2005-12:gross",
                "2007-12:gross",
                "2018-05:row1",
                "2014-09:row3",
                "2019-06:row3",
            }:
                classification = "likely_archived_vs_current_vintage_revision"
                interpretation = (
                    "The frozen archived publication differs from the current "
                    "country-authority IMF history; the archived value is retained."
                )
            else:
                classification = "unclassified_material_reconciliation_difference"
                interpretation = (
                    "Material archived-PDF/current-IMF difference retained for "
                    "manual review; no source is silently overwritten."
                )
            material_reconciliation.append(
                {
                "date": date.date().isoformat(),
                "component": component,
                "pdf_minus_current_imf_musd": float(value),
                    "classification": classification,
                    "interpretation": interpretation,
                }
            )
    metrics = {
        "pdf_target_count": 296,
        "valid_unique_pdf_month_count": int(valid_pdf_count),
        "imf_sdmx_replacement_month_count": int(
            replacement_date in parsed.index
            and str(parsed.loc[replacement_date, "gross_parse_method"]).startswith("IMF")
        ),
        "constructed_month_count": int(len(parsed)),
        "unresolved_month_count": int(296 - len(parsed)),
        "unresolved_failures": [
            {**failure, "date": failure["date"].date().isoformat()}
            for failure in failures
        ],
        "known_archive_wrong_month_links": ["2000-05", "2011-05"],
        "internet_archive_recovered_bcra_months": ["2000-05"],
        "blank_or_absent_assumption_month_count": int(
            parsed["nir_blank_or_absent_assumption_flag"].sum()
        ),
        "component_reporting_quality_counts": {
            component: {
                str(label): int(count)
                for label, count in parsed[
                    f"{component}_reporting_quality"
                ].value_counts(dropna=False).items()
            }
            for component in ("row1", "row2", "row3")
        },
        "na_conservative_variant_nonmissing_month_count": int(
            parsed["nir_strict_nominal_musd_na_conservative"].notna().sum()
        ),
        "archived_pdf_vs_current_imf_material_differences": (
            material_reconciliation
        ),
        "imf_series_status": imf_status,
        "imf_reverse_repo_inflow_diagnostic_musd": {
            date.strftime("%Y-%m"): float(value)
            for date, value in imf["row3_reverse_repo_inflow"].items()
        },
        "reconciliation_materiality_threshold_musd": 0.02,
        "first_parsed_month": parsed.index.min().date().isoformat(),
        "last_parsed_month": parsed.index.max().date().isoformat(),
        "july2024_gross_musd": float(july["nedd_official_reserve_assets_musd"]),
        "july2024_row1_outflows_musd": float(july["row1_outflows"]),
        "july2024_row2_short_musd": float(july["row2_short"]),
        "july2024_row3_outflows_musd": float(july["row3_outflows"]),
        "july2024_strict_nir_nominal_musd": float(july["nir_strict_nominal_musd"]),
        "july2024_narrow_nir_nominal_musd": float(july["nir_narrow_nominal_musd"]),
        "july2024_bcra_gross_strict_nir_nominal_musd": float(
            july["nir_bcra_gross_strict_nominal_musd"]
        ),
        "july2024_us_cpi": base_cpi,
        "chosen_formula": (
            "NEDD official reserve assets + negative gross predetermined monetary-authority "
            "outflows <=12m (row1 principal+interest, row2 shorts, row3 negative details). "
            "This internally consistent NEDD perimeter is primary."
        ),
        "bcra_gross_variant": (
            "BCRA v4 ID1 end-of-month gross + the same NEDD outflows is retained because "
            "the paper says BCRA gross reserves; early levels differ materially from NEDD A."
        ),
        "narrow_formula_status": (
            "robustness only; ID76 is a passive USD repo, not a China-swap series, "
            "and ID1243 has no maturity tag"
        ),
    }
    imf_status_audit = pd.DataFrame(imf_status)
    if not imf_status_audit.empty:
        imf_status_audit.insert(0, "record_type", "imf_series_status")
    parsed_audit = parsed.reset_index()
    parsed_audit.insert(0, "record_type", "constructed_month")
    if not failure_frame.empty:
        failure_frame = failure_frame.copy()
        failure_frame.insert(0, "record_type", "parse_failure")
    parse_audit = pd.concat(
        [
            parsed_audit,
            failure_frame if not failure_frame.empty else pd.DataFrame(),
            imf_status_audit,
        ],
        ignore_index=True,
        sort=False,
    )
    return parsed, metrics, parse_audit


VARIABLE_META: dict[str, dict[str, str]] = {
    "U01": {
        "name": "Argentine monthly CPI inflation",
        "chosen_source": (
            "INDEC historical IPC-GBA through 2006-12; provincial bridge 2007-01; "
            "CIFRA IPC Provincias through 2016-12; INDEC national CPI thereafter"
        ),
        "classification": "close-proxy",
        "confidence": "medium",
        "unit": "percent simple month-on-month",
        "transformation": "100*(linked CPI level / lagged level - 1); log variant retained",
        "revision_status": "mixed; CIFRA is retrospective and official endpoints may be revised",
        "vintage_status": "no consistent real-time vintages",
        "availability_rule": "estimated publication around day 15 of t+1; not exact historically",
    },
    "U02": {
        "name": "Registered nominal wage growth",
        "chosen_source": "INDEC linked no-calificado basic wage/ISBIC; RIPTE from 1994-08 growth",
        "classification": "close-proxy",
        "confidence": "medium",
        "unit": "percent simple month-on-month",
        "transformation": "ratio-link at 1994-07; 100*(level/lagged level-1); log variant retained",
        "revision_status": "conceptual break; provisional RIPTE tail; early history retrospective",
        "vintage_status": "pre-2006 real-time releases not reconstructed",
        "availability_rule": "estimated 45 days after month-end; t-1 often unavailable at day 15",
    },
    "U03": {
        "name": "Monthly economic activity growth, trailing three-month mean",
        "chosen_source": (
            "retrospective PC1 of log real total peso credit (BCRA 23+25)/total "
            "vehicles/crude steel through 1992; linked original NSA EMAE thereafter"
        ),
        "classification": "close-proxy",
        "confidence": "low",
        "unit": "percent simple month-on-month, three-month trailing mean",
        "transformation": "log-level PC1 -> EMAE scale -> splice -> simple growth -> trailing 3M mean",
        "revision_status": "EMAE revisable; PCA is retrospective; 1989-1990 intentionally missing",
        "vintage_status": "no Argentine real-time vintages; pre-1993 proxy not vintage-safe",
        "availability_rule": (
            "illustrative 55-day EMAE lag only; actual calendar required and "
            "t-2 can be unavailable at day 15"
        ),
    },
    "U04": {
        "name": "Nominal effective monthly interest rate",
        "chosen_source": "BCRA code 30 through 2015-11; BCRA v4 ID 160 thereafter",
        "classification": "close-proxy",
        "confidence": "medium",
        "unit": "percent per 30-day month",
        "transformation": "TNA percent * 30/365 (simple nominal monthly convention)",
        "revision_status": "latest-data sources; regime splice",
        "vintage_status": "no historical vintages",
        "availability_rule": "daily/monthly inputs; estimated available by day 10 of t+1",
    },
    "U05": {
        "name": "Monetary-base growth",
        "chosen_source": "BCRA historical code 15, end-of-month stock",
        "classification": "exact-match",
        "confidence": "high",
        "unit": "percent simple month-on-month",
        "transformation": "100*(end-of-month stock/lagged stock-1); log variant retained",
        "revision_status": "latest-data historical file may be corrected",
        "vintage_status": "no historical vintages",
        "availability_rule": "estimated available by day 10 of t+1",
    },
    "U06": {
        "name": "M2-total growth",
        "chosen_source": "BCRA historical code 3543, end-of-month stock",
        "classification": "exact-match",
        "confidence": "high",
        "unit": "percent simple month-on-month",
        "transformation": "100*(end-of-month stock/lagged stock-1); log variant retained",
        "revision_status": "latest-data historical file may be corrected",
        "vintage_status": "no historical vintages",
        "availability_rule": "estimated available by day 10 of t+1",
    },
    "U07": {
        "name": "International wheat-price growth",
        "chosen_source": "World Bank Pink Sheet, Wheat US HRW",
        "classification": "close-proxy",
        "confidence": "medium",
        "unit": "percent simple month-on-month",
        "transformation": "100*(monthly price/lagged price-1); log variant retained",
        "revision_status": "latest workbook may revise history",
        "vintage_status": "no frozen historical releases in this build",
        "availability_rule": "monthly benchmark; estimated day 10 of t+1",
    },
    "U08": {
        "name": "Brent oil-price growth",
        "chosen_source": "World Bank Pink Sheet Brent column; EIA/FRED control",
        "classification": "weak-proxy before 1987; exact/close thereafter",
        "confidence": "medium",
        "unit": "percent simple month-on-month",
        "transformation": "100*(monthly price/lagged price-1); log variant retained",
        "revision_status": "latest workbook; historical backcast before comparable Brent market",
        "vintage_status": "no frozen historical releases in this build",
        "availability_rule": "underlying market known by month-end; workbook published later",
    },
    "U09": {
        "name": "US monthly CPI inflation",
        "chosen_source": "BLS CPI-U NSA via FRED CPIAUCNS",
        "classification": "exact-match",
        "confidence": "high",
        "unit": "percent simple month-on-month",
        "transformation": "100*(CPI-U NSA/lagged CPI-U NSA-1); log variant retained",
        "revision_status": "rare corrections",
        "vintage_status": "latest FRED observations; raw realtime fields retained",
        "availability_rule": "estimated day 14 of t+1",
    },
    "U10": {
        "name": "Official ARS/USD growth",
        "chosen_source": "OECD official monthly average via FRED ARGCCUSMA02STM",
        "classification": "close-proxy",
        "confidence": "medium",
        "unit": "percent simple month-on-month",
        "transformation": "100*(monthly average/lagged monthly average-1); log variant retained",
        "revision_status": "latest-data republished official series",
        "vintage_status": "raw FRED realtime fields retained; no direct BCRA vintages",
        "availability_rule": "underlying daily quotes contemporaneously observable",
    },
    "U11": {
        "name": "Relevant parallel/free-market ARS/USD growth",
        "chosen_source": (
            "regime-dependent BCRA financial/free and official-unified markets; "
            "Ámbito blue 2011-10..2012-12 and CCL in modern control regimes"
        ),
        "classification": "weak-proxy",
        "confidence": "low",
        "unit": "percent simple month-on-month",
        "transformation": "currency-normalize -> regime splice -> monthly mean -> simple growth",
        "revision_status": "Ámbito endpoint undocumented; historical BCRA denomination breaks",
        "vintage_status": "raw endpoint snapshot only; no historical vintages",
        "availability_rule": "underlying quotes contemporaneously observable where series exists",
    },
    "U12": {
        "name": "Parallel-official exchange-rate gap",
        "chosen_source": "derived from U11 and U10 under matched monthly-average convention",
        "classification": "weak-proxy",
        "confidence": "low",
        "unit": "percent",
        "transformation": "100*(parallel level / official level - 1)",
        "revision_status": "inherits both exchange-rate inputs",
        "vintage_status": "inherits both exchange-rate inputs",
        "availability_rule": "contemporaneously computable where both quotes exist",
    },
    "U13": {
        "name": "Real net international reserves",
        "chosen_source": (
            "BCRA NEDD official reserve assets plus gross predetermined monetary-authority "
            "outflows due within 12 months; CPIAUCNS July-2024 dollars"
        ),
        "classification": "close-proxy",
        "confidence": "medium",
        "unit": "million July-2024 US dollars",
        "transformation": (
            "NEDD gross + negative <=12m gross outflow details; explicit "
            "blank/absent-as-zero baseline plus NA-conservative variant; "
            "multiply by CPI_US_2024-07/CPI_US_t"
        ),
        "revision_status": "archived PDF values can be corrected; parser rules frozen in code",
        "vintage_status": "archived PDFs and server Last-Modified retained; BBVA vintages unavailable",
        "availability_rule": "typically day 20-23 of t+1; t-1 not available at day 15",
    },
}


ESTIMATED_LAG_DAYS_FROM_MONTH_END = {
    "U01": 15,
    "U02": 45,
    "U03": 55,
    "U04": 10,
    "U05": 10,
    "U06": 10,
    "U07": 10,
    "U08": 0,
    "U09": 14,
    "U10": 0,
    "U11": 0,
    "U12": 0,
    "U13": 23,
}

VARIABLE_SOURCE_DATASET_PREFIXES = {
    "U01": (
        "indec_178.1_",
        "indec_148.3_",
        "indec_193.1_",
        "indec_195.1_",
        "indec_196.1_",
        "indec_197.1_",
        "cifra_ipc_",
    ),
    "U02": ("indec_158.1_", "indec_historical_wage_", "labor_isbic_"),
    "U03": (
        "bcra_panser",
        "indec_330.3_",
        "indec_359.3_",
        "indec_10.3_",
        "indec_143.3_",
        "official_automotive_",
        "official_steel_",
        "ministry_economic_",
        "indec_industrial_",
        "adefa_yearbook_",
        "indec_178.1_",
        "indec_148.3_",
        "indec_195.1_",
        "indec_196.1_",
        "indec_197.1_",
        "cifra_ipc_",
    ),
    "U04": ("bcra_panser", "bcra_v4_160", "bcra_v4_catalog"),
    "U05": ("bcra_panser",),
    "U06": ("bcra_panser",),
    "U07": ("world_bank_pink_sheet", "fred_PWHEAMTUSDM"),
    "U08": (
        "world_bank_pink_sheet",
        "fred_DCOILBRENTEU",
        "fred_POILBREUSDM",
    ),
    "U09": ("fred_CPIAUCNS",),
    "U10": (
        "fred_ARGCCUSMA02STM",
        "indec_175.1_DR_OFIC",
        "indec_175.1_DR_REFE",
    ),
    "U11": (
        "fred_ARGCCUSMA02STM",
        "indec_175.1_DR_FIN",
        "indec_175.1_DR_LIB",
        "ambito_",
    ),
    "U12": (
        "fred_ARGCCUSMA02STM",
        "indec_175.1_DR_OFIC",
        "indec_175.1_DR_REFE",
        "indec_175.1_DR_FIN",
        "indec_175.1_DR_LIB",
        "ambito_",
    ),
    "U13": (
        "bcra_nedd_",
        "imf_irfcl_",
        "bcra_v4_1",
        "bcra_v4_76",
        "bcra_v4_1243",
        "bcra_v4_catalog",
        "fred_CPIAUCNS",
    ),
}


def _variable_raw_source_map(raw_manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    dataset_ids = raw_manifest["dataset_id"].astype("string")
    keep_columns = [
        "dataset_id",
        "request_id",
        "institution",
        "endpoint",
        "local_path",
        "retrieved_at_utc",
        "sha256",
        "source_last_modified",
        "source_release_date",
        "vintage_policy",
        "observation_start_requested",
        "observation_end_requested",
    ]
    for variable_id, prefixes in VARIABLE_SOURCE_DATASET_PREFIXES.items():
        mask = dataset_ids.str.startswith(prefixes, na=False)
        if variable_id == "U13":
            # `bcra_v4_1` must be an exact match; a prefix match would
            # incorrectly pull the monetary-policy rate (`bcra_v4_160`) into
            # the reserve provenance.
            mask &= ~dataset_ids.eq("bcra_v4_160")
        selected = raw_manifest.loc[mask, keep_columns].copy()
        selected.insert(0, "variable_id", variable_id)
        selected.insert(
            1,
            "source_role",
            np.select(
                [
                    selected["dataset_id"].str.contains(
                        "adefa_yearbook", case=False, na=False
                    ),
                    selected["dataset_id"].str.contains(
                        r"indec_(?:193\.1|195\.1|196\.1|197\.1)|"
                        r"indec_(?:10\.3_ISD|143\.3_NO_PR_2004_A_31)",
                        regex=True,
                        na=False,
                    ),
                    selected["dataset_id"].str.contains(
                        r"codelist|methodology|catalog", case=False, regex=True, na=False
                    ),
                ],
                ["negative_evidence", "bridge_or_sensitivity", "metadata_or_methodology"],
                default="baseline_input_or_validation_control",
            ),
        )
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["variable_id", "request_id", "local_path"]).any():
        raise ValueError("Variable-to-raw-source map contains duplicate keys.")
    return result


def _verify_raw_snapshot(raw: Path) -> pd.DataFrame:
    manifest_path = raw / "retrieval_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    failures: list[str] = []
    for row in manifest.itertuples(index=False):
        path = PROJECT_ROOT / row.local_path
        if not path.is_file():
            failures.append(f"missing:{row.local_path}")
            continue
        actual = sha256_file(path)
        if actual != row.sha256:
            failures.append(f"sha256:{row.local_path}")
        if int(path.stat().st_size) != int(row.byte_count):
            failures.append(f"bytes:{row.local_path}")
    if failures:
        raise ValueError(f"Raw snapshot integrity failure(s): {failures[:20]}")
    return manifest


def _availability_long(
    wide: pd.DataFrame,
    nir: pd.DataFrame,
    *,
    snapshot_id: str,
    retrieved_at_utc: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for variable_id, column in VARIABLE_COLUMNS.items():
        metadata = VARIABLE_META[variable_id]
        frame = pd.DataFrame(
            {
                "date": wide.index,
                "variable_id": variable_id,
                "variable_name": metadata["name"],
                "column_name": column,
                "value": wide[column].to_numpy(),
                "unit": metadata["unit"],
                "chosen_source": metadata["chosen_source"],
                "classification": metadata["classification"],
                "confidence": metadata["confidence"],
                "transformation": metadata["transformation"],
                "revision_status": metadata["revision_status"],
                "vintage_status": metadata["vintage_status"],
                "publication_lag_rule": metadata["availability_rule"],
                "snapshot_id": snapshot_id,
                "retrieved_at_utc": retrieved_at_utc,
                "raw_retrieval_manifest": (
                    f"data/raw/snapshots/{snapshot_id}/retrieval_manifest.csv"
                ),
                "variable_raw_source_map": "audits/variable_raw_source_map.csv",
                "construction_quality_flag": "",
            }
        )
        month_end = frame["date"] + pd.offsets.MonthEnd(1)
        lag = ESTIMATED_LAG_DAYS_FROM_MONTH_END[variable_id]
        frame["estimated_availability_date"] = month_end + pd.to_timedelta(lag, unit="D")
        frame["estimated_availability_date_status"] = "rule_of_thumb_not_actual_release"
        frame["source_release_date"] = pd.NaT
        frame["source_release_date_status"] = "unavailable"
        frame["source_last_modified_date"] = pd.NaT
        if variable_id in {"U08", "U10", "U11", "U12"}:
            frame["estimated_availability_date_status"] = (
                "month_end_information_completion_for_underlying_quotes"
            )
        if variable_id == "U13":
            modified = nir["nedd_source_last_modified"].reindex(frame["date"])
            frame["source_last_modified_date"] = modified.to_numpy()
            assumption = nir[
                "nir_blank_or_absent_assumption_flag"
            ].reindex(frame["date"])
            frame["construction_quality_flag"] = np.where(
                assumption.astype("boolean").fillna(False),
                "blank_or_absent_liability_cell_assumed_zero; "
                "see NA-conservative detail variant",
                "reported_values_or_explicit_zeroes",
            )
            frame["source_release_date_status"] = (
                "first release unavailable; HTTP Last-Modified kept separately and "
                "never treated as publication date"
            )

        forecast_origin = month_end
        mid_next_month = frame["date"] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
        frame["known_by_end_of_month_t"] = np.where(
            frame["value"].isna(),
            "not_available",
            np.where(frame["estimated_availability_date"] <= forecast_origin, "yes", "no"),
        )
        frame["known_by_day15_t_plus_1"] = np.where(
            frame["value"].isna(),
            "not_available",
            np.where(frame["estimated_availability_date"] <= mid_next_month, "yes", "no"),
        )
        if variable_id == "U02":
            retrospective = frame["date"] < pd.Timestamp("2006-01-01")
            frame.loc[retrospective, "known_by_end_of_month_t"] = "not_reconstructable"
            frame.loc[retrospective, "known_by_day15_t_plus_1"] = "not_reconstructable"
        if variable_id == "U03":
            pre_1993 = frame["date"] < pd.Timestamp("1993-01-01")
            pre_1993_observed = pre_1993 & frame["value"].notna()
            frame.loc[pre_1993, "estimated_availability_date"] = pd.NaT
            frame.loc[
                pre_1993_observed, "estimated_availability_date_status"
            ] = "historical_component_release_dates_unavailable"
            frame.loc[
                pre_1993 & frame["value"].isna(),
                "estimated_availability_date_status",
            ] = "not_available"
            frame.loc[
                pre_1993_observed, "known_by_end_of_month_t"
            ] = "not_vintage_safe"
            frame.loc[
                pre_1993_observed, "known_by_day15_t_plus_1"
            ] = "not_vintage_safe"
        if variable_id == "U01":
            retrospective = frame["date"].between("2007-01-01", "2016-12-01")
            frame.loc[retrospective, "known_by_end_of_month_t"] = "not_vintage_safe"
            frame.loc[retrospective, "known_by_day15_t_plus_1"] = "not_vintage_safe"
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    for column in (
        "date",
        "estimated_availability_date",
        "source_release_date",
        "source_last_modified_date",
    ):
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return result


def _complete_blocks(mask: pd.Series) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    start: pd.Timestamp | None = None
    previous: pd.Timestamp | None = None
    for date, complete in mask.items():
        if complete and start is None:
            start = date
        if not complete and start is not None and previous is not None:
            blocks.append(
                {
                    "start": start.date().isoformat(),
                    "end": previous.date().isoformat(),
                    "months": int((previous.to_period("M") - start.to_period("M")).n + 1),
                }
            )
            start = None
        previous = date
    if start is not None and previous is not None:
        blocks.append(
            {
                "start": start.date().isoformat(),
                "end": previous.date().isoformat(),
                "months": int((previous.to_period("M") - start.to_period("M")).n + 1),
            }
        )
    return blocks


def _write_processed_files(
    *,
    snapshot_id: str,
    wide: pd.DataFrame,
    details: pd.DataFrame,
    provenance: pd.DataFrame,
    metrics: dict[str, Any],
    transformations: list[dict[str, Any]],
    audits: dict[str, pd.DataFrame],
) -> Path:
    output = PROCESSED_ROOT / "snapshots" / snapshot_id
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable processed build directory: {output}"
        )
    output.mkdir(parents=True)

    files: dict[str, pd.DataFrame] = {
        "forte_13_public_proxy_monthly_1965_2024.csv": wide,
        "forte_13_construction_details_monthly.csv": details,
        "forte_13_public_proxy_long_with_availability.csv": provenance,
        **{f"audits/{name}": frame for name, frame in audits.items()},
    }
    core8_ids = [
        item
        for item in VARIABLE_COLUMNS
        if item not in {"U03", "U04", "U11", "U12", "U13"}
    ]
    core9_ids = [item for item in VARIABLE_COLUMNS if item not in {"U03", "U11", "U12", "U13"}]
    core10_ids = [item for item in VARIABLE_COLUMNS if item not in {"U11", "U12", "U13"}]
    core8_cols = [VARIABLE_COLUMNS[item] for item in core8_ids]
    core9_cols = [VARIABLE_COLUMNS[item] for item in core9_ids]
    core10_cols = [VARIABLE_COLUMNS[item] for item in core10_ids]
    full_cols = list(VARIABLE_COLUMNS.values())
    files["forte_core8_continuous_complete_monthly.csv"] = wide[core8_cols].dropna()
    files["forte_core9_complete_monthly.csv"] = wide[core9_cols].dropna()
    files["forte_core10_with_activity_complete_segments.csv"] = wide[core10_cols].dropna()
    files["forte_full13_complete_monthly.csv"] = wide[full_cols].dropna()

    file_rows: list[dict[str, Any]] = []
    for filename, frame in files.items():
        target = output / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        index_label = "date" if isinstance(frame.index, pd.DatetimeIndex) else None
        frame.to_csv(target, index=index_label is not None, index_label=index_label)
        file_rows.append(
            {
                "snapshot_id": snapshot_id,
                "file": safe_relpath(target),
                "rows": len(frame),
                "columns": len(frame.columns),
                "sha256": sha256_file(target),
                "created_at_utc": utc_now_iso(),
            }
        )

    transformation_path = output / "transformation_manifest.csv"
    pd.DataFrame(transformations).to_csv(transformation_path, index=False)
    file_rows.append(
        {
            "snapshot_id": snapshot_id,
            "file": safe_relpath(transformation_path),
            "rows": len(transformations),
            "columns": len(pd.DataFrame(transformations).columns),
            "sha256": sha256_file(transformation_path),
            "created_at_utc": utc_now_iso(),
        }
    )
    metrics_path = output / "construction_metrics.json"
    write_json(metrics_path, metrics)
    file_rows.append(
        {
            "snapshot_id": snapshot_id,
            "file": safe_relpath(metrics_path),
            "rows": "",
            "columns": "",
            "sha256": sha256_file(metrics_path),
            "created_at_utc": utc_now_iso(),
        }
    )
    profiles: list[dict[str, Any]] = []
    for label, identifiers in (
        ("core8_continuous_without_activity_interest_parallel_gap_nir", core8_ids),
        ("core9_without_activity_parallel_gap_nir", core9_ids),
        ("core10_with_activity", core10_ids),
        ("full13", list(VARIABLE_COLUMNS)),
    ):
        columns = [VARIABLE_COLUMNS[item] for item in identifiers]
        complete = wide[columns].notna().all(axis=1)
        profiles.append(
            {
                "sample": label,
                "included_variable_ids": ",".join(identifiers),
                "complete_months": int(complete.sum()),
                "first_complete_month": (
                    complete.index[complete][0].date().isoformat() if complete.any() else ""
                ),
                "last_complete_month": (
                    complete.index[complete][-1].date().isoformat() if complete.any() else ""
                ),
                "complete_blocks_json": json.dumps(_complete_blocks(complete)),
            }
        )
    profiles_path = output / "sample_profiles.csv"
    pd.DataFrame(profiles).to_csv(profiles_path, index=False)
    file_rows.append(
        {
            "snapshot_id": snapshot_id,
            "file": safe_relpath(profiles_path),
            "rows": len(profiles),
            "columns": len(pd.DataFrame(profiles).columns),
            "sha256": sha256_file(profiles_path),
            "created_at_utc": utc_now_iso(),
        }
    )
    pd.DataFrame(file_rows).to_csv(output / "processed_file_manifest.csv", index=False)
    return output


def build_dataset(snapshot_id: str | None = None) -> BuildArtifacts:
    ensure_output_dirs()
    selected = snapshot_id or latest_snapshot_id()
    raw = snapshot_path(selected)
    raw_manifest = _verify_raw_snapshot(raw)
    variable_source_map = _variable_raw_source_map(raw_manifest)
    retrieved_at = str(pd.to_datetime(raw_manifest["retrieved_at_utc"]).max())
    panser = _read_bcra_panser(raw)

    cpi, cpi_metrics = construct_cpi(raw)
    wages, wage_metrics, wage_audit = construct_wages(raw)
    (
        activity,
        activity_metrics,
        activity_loadings,
        activity_transcription_audit,
    ) = construct_activity(raw, panser, cpi)
    standard, standard_metrics = construct_standard_variables(raw, panser)
    parallel, parallel_metrics, fx_normalization_audit = construct_parallel_fx(
        raw, standard["official_fx_ars_per_usd"]
    )
    nir, nir_metrics, nir_audit = construct_nir(raw, standard["us_cpi_level"])

    index = month_index()
    wide = pd.DataFrame(index=index)
    wide[VARIABLE_COLUMNS["U01"]] = cpi["arg_cpi_inflation_pct"].reindex(index)
    wide[VARIABLE_COLUMNS["U02"]] = wages["registered_wage_growth_pct"].reindex(index)
    wide[VARIABLE_COLUMNS["U03"]] = activity["activity_growth_3mma_pct"].reindex(index)
    wide[VARIABLE_COLUMNS["U04"]] = standard["nominal_interest_tem_pct"].reindex(index)
    wide[VARIABLE_COLUMNS["U05"]] = standard["monetary_base_growth_pct"].reindex(index)
    wide[VARIABLE_COLUMNS["U06"]] = standard["m2_growth_pct"].reindex(index)
    wide[VARIABLE_COLUMNS["U07"]] = standard["wheat_growth_pct"].reindex(index)
    wide[VARIABLE_COLUMNS["U08"]] = standard["brent_growth_pct"].reindex(index)
    wide[VARIABLE_COLUMNS["U09"]] = standard["us_cpi_inflation_pct"].reindex(index)
    wide[VARIABLE_COLUMNS["U10"]] = standard["official_fx_growth_pct"].reindex(index)
    wide[VARIABLE_COLUMNS["U11"]] = parallel["parallel_fx_growth_pct"].reindex(index)
    wide[VARIABLE_COLUMNS["U12"]] = parallel["fx_gap_pct"].reindex(index)
    wide[VARIABLE_COLUMNS["U13"]] = nir["nir_strict_real_jul2024_musd"].reindex(index)

    details = pd.concat(
        {
            "cpi": cpi.reindex(index),
            "wages": wages.reindex(index),
            "activity": activity.reindex(index),
            "standard": standard.reindex(index),
            "parallel_fx": parallel.reindex(index),
            "nir": nir.reindex(index),
        },
        axis=1,
    )
    details.columns = [f"{group}__{column}" for group, column in details.columns]

    provenance = _availability_long(
        wide, nir, snapshot_id=selected, retrieved_at_utc=retrieved_at
    )
    transformations = [
        {"variable_id": variable_id, "output_column": VARIABLE_COLUMNS[variable_id], **metadata}
        for variable_id, metadata in VARIABLE_META.items()
    ]
    metrics = {
        "snapshot_id": selected,
        "build_created_at_utc": utc_now_iso(),
        "target_start": TARGET_START.date().isoformat(),
        "target_end": TARGET_END.date().isoformat(),
        "raw_request_count": int(len(raw_manifest)),
        "cpi": cpi_metrics,
        "wages": wage_metrics,
        "activity": activity_metrics,
        "standard_variables": standard_metrics,
        "parallel_fx": parallel_metrics,
        "net_international_reserves": nir_metrics,
    }

    output = _write_processed_files(
        snapshot_id=selected,
        wide=wide,
        details=details,
        provenance=provenance,
        metrics=metrics,
        transformations=transformations,
        audits={
            "historical_wage_transcription.csv": wage_audit.reset_index(),
            "activity_pca_loadings.csv": activity_loadings.reset_index(drop=True),
            "activity_archival_transcription.csv": (
                activity_transcription_audit.reset_index(drop=True)
            ),
            "historical_fx_normalization_daily.csv": (
                fx_normalization_audit.reset_index()
            ),
            "nedd_parse_and_imf_reconciliation.csv": nir_audit.reset_index(drop=True),
            "variable_raw_source_map.csv": variable_source_map,
        },
    )
    build_pointer = {
        "snapshot_id": selected,
        "built_at_utc": utc_now_iso(),
        "output_directory": safe_relpath(output),
        "wide_dataset": safe_relpath(output / "forte_13_public_proxy_monthly_1965_2024.csv"),
        "long_dataset": safe_relpath(
            output / "forte_13_public_proxy_long_with_availability.csv"
        ),
        "processed_manifest": safe_relpath(output / "processed_file_manifest.csv"),
    }
    write_json(METADATA_ROOT / "latest_build.json", build_pointer)
    return BuildArtifacts(
        snapshot_id=selected,
        wide=wide,
        details=details,
        provenance=provenance,
        validation_metrics=[],
        transformations=transformations,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the audited 13-variable public proxy.")
    parser.add_argument("--snapshot-id", default=None, help="Raw immutable snapshot to build.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset(args.snapshot_id)
    print(
        f"Built snapshot {result.snapshot_id}: {len(result.wide)} months, "
        f"{len(result.wide.columns)} underlying variables."
    )


if __name__ == "__main__":
    main()
