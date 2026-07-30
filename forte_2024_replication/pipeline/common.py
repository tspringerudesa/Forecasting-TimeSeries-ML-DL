from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "snapshots"
INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
METADATA_ROOT = PROJECT_ROOT / "metadata"
DOCS_ROOT = PROJECT_ROOT / "docs"

TARGET_START = pd.Timestamp("1965-01-01")
TARGET_END = pd.Timestamp("2024-07-01")
TARGET_END_DAY = "2024-07-31"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_snapshot_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_output_dirs() -> None:
    for path in (RAW_ROOT, INTERIM_ROOT, PROCESSED_ROOT, METADATA_ROOT, DOCS_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relpath(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latest_snapshot_id() -> str:
    pointer = METADATA_ROOT / "latest_snapshot.json"
    if not pointer.exists():
        raise FileNotFoundError(
            "No snapshot pointer exists. Run pipeline/fetch_raw.py or pipeline/run_pipeline.py first."
        )
    value = read_json(pointer)
    return str(value["snapshot_id"])


def snapshot_path(snapshot_id: str | None = None) -> Path:
    selected = snapshot_id or latest_snapshot_id()
    path = RAW_ROOT / selected
    if not path.is_dir():
        raise FileNotFoundError(f"Snapshot does not exist: {path}")
    return path


def parse_number_locale(token: str) -> float:
    """Parse either 1,234.56 or 1.234,56 without guessing a currency unit."""

    value = token.strip().replace("\u2212", "-").replace("\xa0", "")
    value = re.sub(r"[^0-9,.\-+]", "", value)
    if not value or value in {"-", "+", ".", ","}:
        return float("nan")
    if "," in value and "." in value:
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
    elif "," in value:
        tail = value.rsplit(",", 1)[1]
        if len(tail) in {1, 2}:
            value = value.replace(",", ".")
        else:
            value = value.replace(",", "")
    elif value.count(".") > 1:
        pieces = value.split(".")
        if len(pieces[-1]) in {1, 2}:
            value = "".join(pieces[:-1]) + "." + pieces[-1]
        else:
            value = "".join(pieces)
    return float(value)


def log_growth(series: pd.Series, name: str | None = None) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").where(lambda x: x > 0)
    result = 100.0 * np.log(clean).diff()
    result.name = name or series.name
    return result


def pct_growth(series: pd.Series, name: str | None = None) -> pd.Series:
    """Conventional simple month-on-month percentage variation."""

    clean = pd.to_numeric(series, errors="coerce").where(lambda x: x > 0)
    result = 100.0 * clean.pct_change(fill_method=None)
    result.name = name or series.name
    return result


def month_index(start: pd.Timestamp = TARGET_START, end: pd.Timestamp = TARGET_END) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="MS", name="date")


def monthly_mean(series: pd.Series) -> pd.Series:
    series = series.copy()
    series.index = pd.to_datetime(series.index)
    return series.sort_index().resample("MS").mean()


def monthly_last(series: pd.Series) -> pd.Series:
    series = series.copy()
    series.index = pd.to_datetime(series.index)
    return series.dropna().sort_index().resample("MS").last()


def require_fred_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is not visible in this process. Run inside the "
            "'timeseriesforecasting' Conda environment; the key is never persisted."
        )
    return key


def redact_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: ("<redacted>" if k.lower() == "api_key" else v) for k, v in params.items()}
