"""Small, read-only API probes for the Forte (2024) source audit.

The script prints compact metadata plus at most two observations per series.
It deliberately does not download full histories and never writes API keys.

Examples
--------
conda run -n timeseriesforecasting python probe_api_metadata.py --provider bcra
conda run -n timeseriesforecasting python probe_api_metadata.py --provider indec
conda run -n timeseriesforecasting python probe_api_metadata.py --provider fred
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import requests


BCRA_IDS = [1, 5, 12, 15, 74, 75, 76, 77, 103, 107, 109, 117, 1200, 1233, 1243]
INDEC_IDS = [
    "178.1_NL_GENERAL_0_0_13",
    "148.3_INIVELNAL_DICI_M_26",
    "145.3_INGNACUAL_DICI_M_38",
    "96.3_INGV_2008_M_20",
    "193.1_NIVEL_GENERAL_JULI_0_13",
    "194.1_NIVEL_GENERAL_2014_0_13",
    "195.1_NIVEL_GENERAL_0_0_13",
    "196.1_NIVEL_GENERAL_2014_0_13",
    "197.1_NIVEL_GENERAL_2014_0_13",
    "143.3_NO_PR_2004_A_21",
    "143.3_NO_PR_2004_A_31",
    "10.3_ISOM_1993_M_29",
    "10.3_ISD_1993_M_31",
    "158.1_REPTE_0_0_5",
    "351.1_BASICO_CONNIO__15",
    "351.2_BASICO_CONNIO__15",
    "330.3_PRODUCCIONLES__22",
    "330.3_PRODUCCIONIOS__22",
    "330.3_PRODUCCIONSTO__16",
    "330.3_PRODUCCIONA_B__22",
    "359.3_ACERO_CRUDUDO__11",
    "175.1_DR_LIBRNTA_0_0_17",
    "175.1_DR_FINANTA_0_0_22",
    "175.1_DR_FINANTA_0_0_21",
    "175.1_DR_OFICNTA_0_0_19",
    "175.1_DR_REFE500_0_0_25",
    "168.1_T_CAMBIOR_D_0_0_26",
    "92.1_RID_0_0_32",
]
FRED_IDS = [
    "CPIAUCNS",
    "ARGCCUSMA02STM",
    "DCOILBRENTEU",
    "POILBREUSDM",
    "PWHEAMTUSDM",
    "WPU0121",
]


def emit(record: dict[str, Any]) -> None:
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))


def probe_bcra(session: requests.Session) -> None:
    catalog_url = "https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias"
    for series_id in BCRA_IDS:
        response = session.get(
            catalog_url,
            params={"IdVariable": series_id, "Limit": 10},
            timeout=30,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            emit({"provider": "BCRA", "series_id": series_id, "verified": False})
            continue
        metadata = results[0]
        samples: list[dict[str, Any]] = []
        for field in ("primerFechaInformada", "ultFechaInformada"):
            raw_date = metadata.get(field)
            if not raw_date:
                continue
            date = raw_date[:10]
            detail_response = session.get(
                f"{catalog_url}/{series_id}",
                params={"Desde": date, "Hasta": date, "Limit": 2},
                timeout=30,
            )
            detail_response.raise_for_status()
            detail = detail_response.json().get("results", [{}])[0].get("detalle", [])
            samples.extend(detail[:1])
        emit(
            {
                "provider": "BCRA",
                "series_id": series_id,
                "verified": bool(samples),
                "metadata": {
                    key: metadata.get(key)
                    for key in (
                        "descripcion",
                        "categoria",
                        "tipoSerie",
                        "periodicidad",
                        "unidadExpresion",
                        "moneda",
                        "primerFechaInformada",
                        "ultFechaInformada",
                    )
                },
                "samples": samples,
            }
        )


def probe_indec(session: requests.Session) -> None:
    url = "https://apis.datos.gob.ar/series/api/series"
    for series_id in INDEC_IDS:
        endpoint_samples = []
        payload: dict[str, Any] = {}
        for sort in ("asc", "desc"):
            response = session.get(
                url,
                params={
                    "ids": series_id,
                    "metadata": "full",
                    "limit": 1,
                    "sort": sort,
                },
                timeout=30,
            )
            response.raise_for_status()
            current_payload = response.json()
            if not payload:
                payload = current_payload
            endpoint_samples.extend(current_payload.get("data", [])[:1])
        meta = payload.get("meta", [])
        global_meta = meta[0] if meta else {}
        series_meta = meta[1] if len(meta) > 1 else {}
        emit(
            {
                "provider": "Series de Tiempo Argentina",
                "series_id": series_id,
                "verified": bool(payload.get("data")),
                "global": {
                    key: global_meta.get(key)
                    for key in ("frequency", "start_date", "end_date")
                },
                "field": {
                    key: series_meta.get("field", {}).get(key)
                    for key in ("id", "description", "units", "frequency")
                },
                "dataset": {
                    key: series_meta.get("dataset", {}).get(key)
                    for key in ("title", "source", "issued")
                },
                "distribution_url": series_meta.get("distribution", {}).get("downloadURL"),
                "confirmed_first": endpoint_samples[0] if endpoint_samples else None,
                "confirmed_last": endpoint_samples[-1] if endpoint_samples else None,
            }
        )


def probe_fred(session: requests.Session) -> None:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise SystemExit("FRED_API_KEY is not set in this process.")
    base = "https://api.stlouisfed.org/fred"
    for series_id in FRED_IDS:
        common = {"api_key": api_key, "file_type": "json", "series_id": series_id}
        metadata_response = session.get(f"{base}/series", params=common, timeout=30)
        if not metadata_response.ok:
            emit(
                {
                    "provider": "FRED",
                    "series_id": series_id,
                    "verified": False,
                    "http_status": metadata_response.status_code,
                }
            )
            continue
        metadata = metadata_response.json().get("seriess", [{}])[0]
        sample_response = session.get(
            f"{base}/series/observations",
            params={
                **common,
                "observation_start": metadata.get("observation_start"),
                "observation_end": metadata.get("observation_start"),
                "limit": 2,
            },
            timeout=30,
        )
        sample_response.raise_for_status()
        samples = sample_response.json().get("observations", [])[:2]
        emit(
            {
                "provider": "FRED",
                "series_id": series_id,
                "verified": any(item.get("value") != "." for item in samples),
                "metadata": {
                    key: metadata.get(key)
                    for key in (
                        "title",
                        "observation_start",
                        "observation_end",
                        "frequency",
                        "units",
                        "seasonal_adjustment",
                        "last_updated",
                        "notes",
                    )
                },
                "samples": samples,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=("bcra", "indec", "fred"))
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = "forte-2024-replication-source-audit/0.1"
    if args.provider == "fred":
        probe_fred(session)
    else:
        {"bcra": probe_bcra, "indec": probe_indec}[args.provider](session)


if __name__ == "__main__":
    main()
