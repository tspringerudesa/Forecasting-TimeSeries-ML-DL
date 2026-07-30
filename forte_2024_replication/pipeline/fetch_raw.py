from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from pipeline.common import (
    METADATA_ROOT,
    RAW_ROOT,
    TARGET_END_DAY,
    default_snapshot_id,
    ensure_output_dirs,
    redact_params,
    safe_relpath,
    sha256_bytes,
    utc_now_iso,
    write_csv_rows,
    write_json,
)


MANIFEST_FIELDS = [
    "snapshot_id",
    "dataset_id",
    "request_id",
    "institution",
    "endpoint",
    "requested_params_json",
    "requested_headers_json",
    "local_path",
    "retrieved_at_utc",
    "http_status",
    "content_type",
    "byte_count",
    "sha256",
    "source_last_modified",
    "source_release_date",
    "vintage_policy",
    "observation_start_requested",
    "observation_end_requested",
    "notes",
]


STATIC_SOURCES = [
    {
        "dataset_id": "bcra_panser",
        "institution": "BCRA",
        "url": "https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/panser.txt",
        "path": "bcra/panser.txt",
        "vintage": "latest-data file; historical revisions overwrite prior values",
        "notes": (
            "Codes 15, 23, 25, 30 and 3543 are selected downstream; code 22 is "
            "retained as a rejected mixed-currency activity-credit alternative."
        ),
    },
    {
        "dataset_id": "bcra_series_catalog",
        "institution": "BCRA",
        "url": "https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/Series_estadisticas.xlsx",
        "path": "bcra/Series_estadisticas.xlsx",
        "vintage": "catalog snapshot at retrieval",
        "notes": "Series definitions and units.",
    },
    {
        "dataset_id": "bcra_historical_cpi",
        "institution": "BCRA",
        "url": "https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/ES_INFO_SERIES_TASHIS.TXT",
        "path": "bcra/ES_INFO_SERIES_TASHIS.TXT",
        "vintage": "latest-data file; historical reconstruction",
        "notes": "Code 7931 is retained as an official CPI cross-check, not the baseline intervention series.",
    },
    {
        "dataset_id": "cifra_ipc_provincias",
        "institution": "CIFRA-CTA",
        "url": "https://centrocifra.org.ar/wp-content/uploads/2023/08/IPC-Provincias-2007-2018.xlsx",
        "path": "cifra/IPC-Provincias-2007-2018.xlsx",
        "vintage": "retrospective reconstruction; no real-time vintages",
        "notes": "Institutional intervention-period CPI proxy.",
    },
    {
        "dataset_id": "cifra_ipc_methodology",
        "institution": "CIFRA-CTA",
        "url": "https://centrocifra.org.ar/wp-content/uploads/2023/08/Nota-metodologica-IPC-Provincias.pdf",
        "path": "cifra/Nota-metodologica-IPC-Provincias.pdf",
        "vintage": "methodology snapshot",
        "notes": "Explains changing provincial composition and imputation.",
    },
    {
        "dataset_id": "indec_historical_wage_volume",
        "institution": "INDEC",
        "url": "https://biblioteca.indec.gob.ar/bases/minde/4si9_4.pdf",
        "path": "wages/4si9_4.pdf",
        "vintage": "1991 archival publication",
        "notes": "Official monthly linked basic-agreement wage tables; OCR text is audited downstream.",
    },
    {
        "dataset_id": "labor_isbic_table",
        "institution": "Secretaría de Seguridad Social",
        "url": "https://www.argentina.gob.ar/sites/default/files/indice_isbic.pdf",
        "path": "wages/indice_isbic.pdf",
        "vintage": "September 2016 publication; definitive through 2015-08, provisional tail",
        "notes": "Qualified and non-qualified monthly ISBIC columns.",
    },
    {
        "dataset_id": "labor_isbic_methodology",
        "institution": "Secretaría de Seguridad Social",
        "url": "https://www.argentina.gob.ar/sites/default/files/informe_isbic.pdf",
        "path": "wages/informe_isbic.pdf",
        "vintage": "methodology snapshot",
        "notes": "ISBIC definitions and comparability limitations.",
    },
    {
        "dataset_id": "world_bank_pink_sheet",
        "institution": "World Bank",
        "url": (
            "https://thedocs.worldbank.org/en/doc/"
            "74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/"
            "CMO-Historical-Data-Monthly.xlsx"
        ),
        "path": "world_bank/CMO-Historical-Data-Monthly.xlsx",
        "vintage": "latest-data workbook; release date embedded in workbook",
        "notes": "Wheat US HRW and Crude oil, Brent columns.",
    },
    {
        "dataset_id": "official_automotive_history_csv",
        "institution": "Secretaría de Política Económica / ADEFA",
        "url": (
            "https://infra.datos.gob.ar/catalog/sspm/dataset/330/distribution/330.3/"
            "download/datos-historicos-industria-automotriz-unidades-mensuales.csv"
        ),
        "path": "activity/datos-historicos-industria-automotriz-unidades-mensuales.csv",
        "vintage": "latest-data official historical file; revisions overwrite prior values",
        "notes": "Primary-file check for the API automobile series and its 1989-1992 null block.",
    },
    {
        "dataset_id": "official_steel_history_csv",
        "institution": "Secretaría de Política Económica / Argentine Steel Chamber",
        "url": (
            "https://infra.datos.gob.ar/catalog/sspm/dataset/359/distribution/359.3/"
            "download/datos-historicos-de-la-industria-siderurgica-datos-mensulaes.csv"
        ),
        "path": "activity/datos-historicos-industria-siderurgica-datos-mensuales.csv",
        "vintage": "latest-data official historical file; revisions overwrite prior values",
        "notes": "Primary-file check for crude steel and its 1991-1992 null block.",
    },
    {
        "dataset_id": "ministry_economic_report_1992",
        "institution": "Ministerio de Economía",
        "url": (
            "http://cdi.mecon.gov.ar/greenstone/collect/iet/index/assoc/"
            "HASH0b19.dir/doc.pdf"
        ),
        "path": "activity/informe_economico_1992.pdf",
        "vintage": "archival publication",
        "notes": (
            "Printed tables AII-4/AII-5 contain monthly 1991 crude-steel and "
            "total-motor-vehicle observations transcribed downstream."
        ),
    },
    {
        "dataset_id": "indec_industrial_products_1993_12",
        "institution": "INDEC",
        "url": "https://biblioteca.indec.gob.ar/bases/minde/epidic93.pdf",
        "path": "activity/estadisticas_productos_industriales_1993_12.pdf",
        "vintage": "December 1993 archival publication",
        "notes": (
            "Revised monthly 1992 crude-steel and motor-vehicle tables, with "
            "industry-source footnotes."
        ),
    },
    {
        "dataset_id": "adefa_yearbook_1989",
        "institution": "ADEFA",
        "url": "https://www.adefa.org.ar/upload/anuarios/anuarios_old/1989.pdf",
        "path": "activity/adefa_1989.pdf",
        "vintage": "archival industry publication",
        "notes": (
            "Negative-evidence archive: contains annual/company/model tables but "
            "does not supply a reproducible complete monthly 1989 total."
        ),
    },
    {
        "dataset_id": "adefa_yearbook_1990",
        "institution": "ADEFA",
        "url": "https://www.adefa.org.ar/upload/anuarios/anuarios_old/1990.pdf",
        "path": "activity/adefa_1990.pdf",
        "vintage": "archival industry publication",
        "notes": (
            "Negative-evidence archive: contains annual/company/model tables but "
            "does not supply a reproducible complete monthly 1990 total."
        ),
    },
    {
        "dataset_id": "bcra_nedd_index",
        "institution": "BCRA",
        "url": "https://www.bcra.gob.ar/normas-especiales-para-la-divulgacion-de-datos-fmi/",
        "path": "nedd/index.html",
        "vintage": "live archive index snapshot",
        "notes": "Official monthly reserves/foreign-currency-liquidity archive.",
    },
    {
        "dataset_id": "bcra_nedd_2000_archive_index",
        "institution": "BCRA via Internet Archive",
        "url": (
            "https://web.archive.org/web/20000819134455id_/"
            "http://www.bcra.gov.ar:80/english/contad/econ0100.htm"
        ),
        "path": "nedd/archive_index_2000.html",
        "vintage": "fixed Internet Archive capture dated 2000-08-19",
        "notes": (
            "Archived original BCRA index that links itemp0500.PDF as the "
            "May-2000 reserves/foreign-currency-liquidity template."
        ),
    },
    {
        "dataset_id": "imf_irfcl_indicator_codelist",
        "institution": "International Monetary Fund",
        "url": (
            "https://api.imf.org/external/sdmx/3.0/structure/codelist/"
            "IMF.STA/CL_IRFCL_INDICATOR_PUB/4.0.0"
            "?detail=full&references=none"
        ),
        "path": "imf_irfcl/CL_IRFCL_INDICATOR_PUB_4.0.0.xml",
        "vintage": "current official IMF SDMX structure snapshot",
        "notes": (
            "Official full IRFCL indicator codelist; used to verify the exact "
            "indicator IDs and labels."
        ),
        "headers": {
            "Accept": "application/vnd.sdmx.structure+xml;version=3.0.0"
        },
    },
]


INDEC_SERIES = {
    "178.1_NL_GENERAL_0_0_13": ("1943-01-01", "historical CPI-GBA linked level"),
    "148.3_INIVELNAL_DICI_M_26": ("2016-12-01", "national CPI level, Dec-2016 base"),
    "195.1_NIVEL_GENERAL_0_0_13": ("2006-12-01", "Mendoza CPI bridge/sensitivity"),
    "196.1_NIVEL_GENERAL_2014_0_13": ("2006-12-01", "Neuquén CPI bridge/sensitivity"),
    "197.1_NIVEL_GENERAL_2014_0_13": ("2006-12-01", "San Luis CPI bridge/sensitivity"),
    "193.1_NIVEL_GENERAL_JULI_0_13": ("2012-07-01", "CABA CPI sensitivity"),
    "158.1_REPTE_0_0_5": ("1994-07-01", "RIPTE"),
    "330.3_PRODUCCIONLES__22": ("1963-01-01", "automobile production"),
    "330.3_PRODUCCIONIOS__22": ("1963-01-01", "utility vehicle production"),
    "330.3_PRODUCCIONSTO__16": (
        "1963-01-01",
        "remaining commercial-vehicle production; preferred total component",
    ),
    "359.3_ACERO_CRUDUDO__11": ("1965-01-01", "raw steel production"),
    "10.3_ISOM_1993_M_29": ("1993-01-01", "EMAE base-1993 original"),
    "10.3_ISD_1993_M_31": ("1993-01-01", "EMAE base-1993 seasonally adjusted"),
    "143.3_NO_PR_2004_A_21": ("2004-01-01", "EMAE base-2004 original"),
    "143.3_NO_PR_2004_A_31": ("2004-01-01", "EMAE base-2004 seasonally adjusted"),
    "175.1_DR_FINANTA_0_0_22": ("1971-01-01", "historical financial USD sale rate"),
    "175.1_DR_LIBRNTA_0_0_17": ("1976-01-01", "historical free USD sale rate"),
    "175.1_DR_OFICNTA_0_0_19": ("1976-01-01", "historical official USD sale rate control"),
    "175.1_DR_REFE500_0_0_25": ("2002-03-01", "modern A3500 reference-rate control"),
}


FRED_SERIES = {
    "CPIAUCNS": "US CPI-U NSA",
    "ARGCCUSMA02STM": "Argentina official monthly exchange rate",
    "DCOILBRENTEU": "Brent daily EIA control",
    "PWHEAMTUSDM": "IMF wheat monthly control",
    "POILBREUSDM": "IMF Brent monthly control",
}


BCRA_V4_SERIES = {
    1: ("1996-01-01", "gross international reserves"),
    76: ("2003-01-01", "foreign-currency passive repo with exterior"),
    1243: ("2003-01-01", "FX current accounts at BCRA"),
    160: ("2015-12-01", "monetary-policy interest rate"),
}

IMF_IRFCL_SERIES = {
    "IRFCLDT1_IRFCL65_USD": "Section I.A official reserve assets",
    "IRFCLDT2_IRFCL80_FO_USD": "Section II category 1 principal outflows, total maturity",
    "IRFCLDT2_IRFCL79_FO_USD": "Section II category 1 interest outflows, total maturity",
    "IRFCLDT2_IRFCL1T_SHP_USD": "Section II gross short forward/futures positions",
    "IRFCLDT2_IRFCL48T_FO_USD": "Section II repo outflows, total maturity",
    "IRFCLDT2_IRFCL50T_FO_USD": "Section II trade-credit outflows, total maturity",
    "IRFCLDT2_IRFCL46T_FO_USD": "Section II other-accounts-payable outflows, total maturity",
    "IRFCLDT2_IRFCL49T_IN_USD": "Section II reverse-repo inflows diagnostic",
    "IRFCLDT2_IRFCL50T_IN_USD": "Section II trade-credit inflows diagnostic",
    "IRFCLDT2_IRFCL47T_IN_USD": "Section II other-accounts-receivable inflows diagnostic",
    "IRFCLDT2_IRFCL78_USD": "Section II category 1 aggregate cross-check",
    "IRFCLDT2_IRFCL85_USD": "Section II category 3 aggregate cross-check",
}

# Argentina does not publish observations for every valid IRFCL detail code.
# The IMF API returns a structurally valid, empty DataSet (rather than a
# not-found response) for these series over the audited window.  Their IDs are
# still independently verified against the official IMF codelist above.
IMF_IRFCL_OPTIONAL_EMPTY = {
    "IRFCLDT2_IRFCL46T_FO_USD",
    "IRFCLDT2_IRFCL50T_IN_USD",
    "IRFCLDT2_IRFCL47T_IN_USD",
}


INDEC_REQUIRED_NON_NULL_DATES = {
    "178.1_NL_GENERAL_0_0_13": {"1965-01-01", "2006-12-01"},
    "148.3_INIVELNAL_DICI_M_26": {"2016-12-01", "2024-07-01"},
    "195.1_NIVEL_GENERAL_0_0_13": {"2006-12-01", "2007-01-01"},
    "196.1_NIVEL_GENERAL_2014_0_13": {"2006-12-01", "2007-01-01"},
    "197.1_NIVEL_GENERAL_2014_0_13": {"2006-12-01", "2007-01-01"},
    "193.1_NIVEL_GENERAL_JULI_0_13": {"2012-07-01"},
    "158.1_REPTE_0_0_5": {"1994-07-01", "1994-08-01", "2024-07-01"},
    "330.3_PRODUCCIONLES__22": {"1965-01-01", "1988-12-01", "1993-01-01"},
    "330.3_PRODUCCIONIOS__22": {"1965-01-01", "1988-12-01", "1993-01-01"},
    "330.3_PRODUCCIONSTO__16": {"1965-01-01", "1988-12-01", "1993-01-01"},
    "359.3_ACERO_CRUDUDO__11": {"1965-01-01", "1990-12-01", "1993-01-01"},
    "10.3_ISOM_1993_M_29": {"1993-01-01", "2013-12-01"},
    "10.3_ISD_1993_M_31": {"1993-01-01", "2013-12-01"},
    "143.3_NO_PR_2004_A_21": {"2004-01-01", "2024-07-01"},
    "143.3_NO_PR_2004_A_31": {"2004-01-01", "2024-07-01"},
    "175.1_DR_FINANTA_0_0_22": {"1971-09-20", "1981-06-22", "1982-07-06"},
    "175.1_DR_LIBRNTA_0_0_17": {"1976-01-08", "1983-01-03", "1987-11-02"},
    "175.1_DR_OFICNTA_0_0_19": {"1976-03-08", "1989-12-19"},
    "175.1_DR_REFE500_0_0_25": {"2002-03-04", "2024-07-31"},
}


@dataclass
class SnapshotFetcher:
    snapshot_id: str
    raw_dir: Path
    session: requests.Session = field(default_factory=requests.Session)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/126 Safari/537.36 "
                    "ForteReplicationAudit/1.0"
                ),
                "Referer": "https://www.ambito.com/",
            }
        )

    def get(
        self,
        *,
        dataset_id: str,
        request_id: str,
        institution: str,
        url: str,
        relative_path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        source_release_date: str = "",
        vintage_policy: str = "",
        observation_start: str = "",
        observation_end: str = "",
        notes: str = "",
        timeout: int = 90,
        retries: int = 4,
    ) -> bytes:
        target = self.raw_dir / relative_path
        if target.exists():
            raise FileExistsError(f"Immutable snapshot target already exists: {target}")
        last_error: Exception | None = None
        response: requests.Response | None = None
        for attempt in range(retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )
                if response.ok:
                    break
                response.raise_for_status()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(1.5 * (attempt + 1))
        if response is None or not response.ok:
            raise RuntimeError(f"Failed after {retries} attempts: {url}") from last_error

        content = response.content
        suffix = Path(relative_path).suffix.lower()
        if suffix == ".pdf" and not content.lstrip().startswith(b"%PDF-"):
            raise ValueError(f"Expected PDF payload but received different bytes: {url}")
        if suffix == ".xlsx" and not content.startswith(b"PK"):
            raise ValueError(f"Expected XLSX/ZIP payload but received different bytes: {url}")
        if suffix == ".json":
            try:
                json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"Expected JSON payload but response is invalid: {url}") from exc
        if suffix == ".xml" and (
            not content.lstrip().startswith(b"<")
            or b"<html" in content[:1000].lower()
        ):
            raise ValueError(f"Expected XML payload but received different bytes: {url}")
        if suffix in {".csv", ".txt"} and b"<html" in content[:1000].lower():
            raise ValueError(f"Expected data payload but received HTML: {url}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        retrieved = utc_now_iso()
        last_modified = response.headers.get("Last-Modified", "")
        self.rows.append(
            {
                "snapshot_id": self.snapshot_id,
                "dataset_id": dataset_id,
                "request_id": request_id,
                "institution": institution,
                "endpoint": url,
                "requested_params_json": json.dumps(
                    redact_params(params or {}), ensure_ascii=False, sort_keys=True
                ),
                "requested_headers_json": json.dumps(
                    headers or {}, ensure_ascii=False, sort_keys=True
                ),
                "local_path": safe_relpath(target),
                "retrieved_at_utc": retrieved,
                "http_status": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "byte_count": len(content),
                "sha256": sha256_bytes(content),
                "source_last_modified": last_modified,
                "source_release_date": source_release_date,
                "vintage_policy": vintage_policy,
                "observation_start_requested": observation_start,
                "observation_end_requested": observation_end,
                "notes": notes,
            }
        )
        return content

    def copy_verified_local_fallback(
        self,
        *,
        dataset_id: str,
        request_id: str,
        institution: str,
        original_endpoint: str,
        fallback_source: Path,
        relative_path: str,
        expected_sha256: str,
        vintage_policy: str,
        observation_start: str,
        observation_end: str,
        notes: str,
    ) -> bytes:
        """Carry a pinned payload forward when its archival carrier is unavailable."""

        target = self.raw_dir / relative_path
        if target.exists():
            raise FileExistsError(f"Immutable snapshot target already exists: {target}")
        if not fallback_source.is_file():
            raise FileNotFoundError(
                "Verified local fallback is unavailable after remote retrieval "
                f"failure: {fallback_source}"
            )
        content = fallback_source.read_bytes()
        observed_sha256 = sha256_bytes(content)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                "Verified local fallback hash mismatch: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        self.rows.append(
            {
                "snapshot_id": self.snapshot_id,
                "dataset_id": dataset_id,
                "request_id": request_id,
                "institution": institution,
                "endpoint": original_endpoint,
                "requested_params_json": "{}",
                "requested_headers_json": "{}",
                "local_path": safe_relpath(target),
                "retrieved_at_utc": utc_now_iso(),
                "http_status": "verified_local_fallback",
                "content_type": "application/pdf",
                "byte_count": len(content),
                "sha256": observed_sha256,
                "source_last_modified": "",
                "source_release_date": "",
                "vintage_policy": vintage_policy,
                "observation_start_requested": observation_start,
                "observation_end_requested": observation_end,
                "notes": (
                    f"{notes} Remote archival carrier was unavailable; bytes "
                    f"were carried from {safe_relpath(fallback_source)} after "
                    "exact SHA-256 verification."
                ),
            }
        )
        return content


def fetch_static(fetcher: SnapshotFetcher) -> None:
    for source in STATIC_SOURCES:
        fetcher.get(
            dataset_id=source["dataset_id"],
            request_id=source["dataset_id"],
            institution=source["institution"],
            url=source["url"],
            relative_path=source["path"],
            headers=source.get("headers"),
            vintage_policy=source["vintage"],
            notes=source["notes"],
        )


def fetch_indec(fetcher: SnapshotFetcher) -> None:
    endpoint = "https://apis.datos.gob.ar/series/api/series"
    for series_id, (start, description) in INDEC_SERIES.items():
        offset = 0
        page = 1
        total = None
        observations: list[list[Any]] = []
        page_size = 1000
        while total is None or offset < total:
            params: dict[str, Any] = {
                "ids": series_id,
                "start_date": start,
                "end_date": TARGET_END_DAY,
                "limit": page_size,
                "start": offset,
                "metadata": "full" if page == 1 else "none",
            }
            content = fetcher.get(
                dataset_id=f"indec_{series_id}",
                request_id=f"{series_id}_page_{page:03d}",
                institution="Datos Argentina / source institution in metadata",
                url=endpoint,
                relative_path=f"indec/{series_id}_page_{page:03d}.json",
                params=params,
                vintage_policy="latest-data API; no historical vintages",
                observation_start=start,
                observation_end=TARGET_END_DAY,
                notes=description,
            )
            payload = json.loads(content)
            total = int(payload.get("count", 0))
            received_rows = payload.get("data", [])
            observations.extend(received_rows)
            received = len(received_rows)
            # API `count`/`start` include null calendar slots, whereas `data`
            # can omit them. Advancing by received rows skips later non-null
            # blocks after a long gap; advance by the requested page window.
            offset += page_size
            page += 1
        if not observations:
            raise RuntimeError(f"INDEC ID {series_id} returned no observations.")
        value_by_date = {
            str(row[0])[:10]: row[1]
            for row in observations
            if isinstance(row, list) and len(row) >= 2
        }
        missing_anchors = [
            date
            for date in INDEC_REQUIRED_NON_NULL_DATES.get(series_id, set())
            if value_by_date.get(date) is None
        ]
        if missing_anchors:
            raise RuntimeError(
                f"INDEC ID {series_id} failed required non-null anchors: {missing_anchors}"
            )


def fetch_fred(fetcher: SnapshotFetcher, api_key: str) -> None:
    for series_id, description in FRED_SERIES.items():
        metadata_params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        }
        fetcher.get(
            dataset_id=f"fred_{series_id}",
            request_id=f"{series_id}_metadata",
            institution="Federal Reserve Bank of St. Louis / source institution",
            url="https://api.stlouisfed.org/fred/series",
            relative_path=f"fred/{series_id}_metadata.json",
            params=metadata_params,
            vintage_policy="FRED metadata snapshot; ALFRED availability varies by source",
            notes=description,
        )
        observations_params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": "1959-01-01",
            "observation_end": TARGET_END_DAY,
        }
        fetcher.get(
            dataset_id=f"fred_{series_id}",
            request_id=f"{series_id}_observations",
            institution="Federal Reserve Bank of St. Louis / source institution",
            url="https://api.stlouisfed.org/fred/series/observations",
            relative_path=f"fred/{series_id}_observations.json",
            params=observations_params,
            vintage_policy="latest-data observation snapshot; realtime fields retained in raw response",
            observation_start="1959-01-01",
            observation_end=TARGET_END_DAY,
            notes=description,
        )


def fetch_bcra_v4(fetcher: SnapshotFetcher) -> None:
    endpoint = "https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias"
    for offset in (0, 1000):
        fetcher.get(
            dataset_id="bcra_v4_catalog",
            request_id=f"catalog_offset_{offset}",
            institution="BCRA",
            url=endpoint,
            relative_path=f"bcra_v4/catalog_offset_{offset:04d}.json",
            params={"Limit": 1000, "Offset": offset},
            vintage_policy="live catalog snapshot",
            notes="Required to verify IDs 1, 76, 1243 and 160.",
        )
    for series_id, (start, description) in BCRA_V4_SERIES.items():
        offset = 0
        page = 1
        total = None
        while total is None or offset < total:
            params = {
                "Desde": start,
                "Hasta": TARGET_END_DAY,
                "Limit": 1000,
                "Offset": offset,
            }
            content = fetcher.get(
                dataset_id=f"bcra_v4_{series_id}",
                request_id=f"{series_id}_page_{page:03d}",
                institution="BCRA",
                url=f"{endpoint}/{series_id}",
                relative_path=f"bcra_v4/{series_id}_page_{page:03d}.json",
                params=params,
                vintage_policy="latest-data API; historical revisions possible; no public vintages",
                observation_start=start,
                observation_end=TARGET_END_DAY,
                notes=description,
            )
            payload = json.loads(content)
            resultset = payload.get("metadata", {}).get("resultset", {})
            total = int(resultset.get("count", 0))
            details = []
            for result in payload.get("results", []):
                details.extend(result.get("detalle", []))
            if not details:
                break
            offset += len(details)
            page += 1


def fetch_imf_irfcl(fetcher: SnapshotFetcher) -> None:
    codelist_path = (
        fetcher.raw_dir / "imf_irfcl" / "CL_IRFCL_INDICATOR_PUB_4.0.0.xml"
    )
    codelist = codelist_path.read_bytes()
    codelist_root = ET.fromstring(codelist)
    codelist_codes = {
        element.attrib["id"]
        for element in codelist_root.iter()
        if element.tag.rsplit("}", 1)[-1] == "Code" and "id" in element.attrib
    }
    missing_codes = [
        code for code in IMF_IRFCL_SERIES if code not in codelist_codes
    ]
    if missing_codes:
        raise RuntimeError(
            "Official IMF IRFCL codelist does not contain required IDs: "
            f"{missing_codes}"
        )
    base = "https://api.imf.org/external/sdmx/2.1/data/IRFCL"
    for code, description in IMF_IRFCL_SERIES.items():
        content = fetcher.get(
            dataset_id=f"imf_irfcl_{code}",
            request_id=code,
            institution="IMF / country authorities",
            url=f"{base}/ARG.{code}.S1X.M",
            relative_path=f"imf_irfcl/{code}.xml",
            params={"startPeriod": "1999-12", "endPeriod": "2024-07"},
            vintage_policy="current IMF SDMX snapshot; not a real-time vintage archive",
            observation_start="1999-12-01",
            observation_end="2024-07-31",
            notes=(
                f"{description}. Institutional cross-check and audited replacement "
                "only where the BCRA archive serves a wrong-month PDF."
            ),
        )
        has_observations = (
            f'INDICATOR="{code}"'.encode("utf-8") in content
            and b"OBS_VALUE" in content
        )
        is_valid_empty_response = (
            code in IMF_IRFCL_OPTIONAL_EMPTY
            and b"StructureSpecificData" in content
            and b"DataSet" in content
            and b"Error" not in content
        )
        if not (has_observations or is_valid_empty_response):
            raise RuntimeError(
                f"IMF IRFCL test fetch did not return observations for {code}."
            )


def fetch_ambito(fetcher: SnapshotFetcher) -> None:
    for kind, endpoint, first_year in (
        ("blue", "https://mercados.ambito.com/dolar/informal/historico-general", 2002),
        ("ccl", "https://mercados.ambito.com/dolarrava/cl/historico-general", 2013),
    ):
        for year in range(first_year, 2025):
            inclusive_end = "2024-07-31" if year == 2024 else f"{year}-12-31"
            # Endpoint end-date is exclusive.
            request_end = "2024-08-01" if year == 2024 else f"{year + 1}-01-01"
            start = f"{year}-01-01"
            content = fetcher.get(
                dataset_id=f"ambito_{kind}",
                request_id=f"{kind}_{year}",
                institution="Ámbito Financiero",
                url=f"{endpoint}/{start}/{request_end}",
                relative_path=f"ambito/{kind}_{year}.json",
                vintage_policy="undocumented current-history endpoint; no vintages",
                observation_start=start,
                observation_end=inclusive_end,
                notes=(
                    "Unofficial fallback. Blue uses sale quote; CCL uses reference quote. "
                    "Endpoint end is exclusive; raw JSON is archived because it can change."
                ),
            )
            payload = json.loads(content)
            if not isinstance(payload, list) or len(payload) < 2:
                raise RuntimeError(f"Ámbito {kind} {year} returned no observation rows.")
            returned_dates = [
                time.strptime(str(row[0]), "%d/%m/%Y")
                for row in payload[1:]
                if isinstance(row, list) and row
            ]
            if not returned_dates:
                raise RuntimeError(f"Ámbito {kind} {year} has no parseable dates.")
            maximum = max(returned_dates)
            expected_month = 7 if year == 2024 else 12
            if maximum.tm_year != year or maximum.tm_mon != expected_month:
                raise RuntimeError(
                    f"Ámbito {kind} {year} unexpectedly ends "
                    f"{maximum.tm_year:04d}-{maximum.tm_mon:02d}-{maximum.tm_mday:02d}."
                )


MONTHS_ES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def fetch_nedd_pdfs(fetcher: SnapshotFetcher) -> None:
    index_path = fetcher.raw_dir / "nedd" / "index.html"
    html = index_path.read_text(encoding="utf-8", errors="replace")
    objects: list[dict[str, str]] = []
    for match in re.findall(r'\{[^{}]*?"url_es"[^{}]*?\}', html):
        try:
            value = json.loads(match)
        except json.JSONDecodeError:
            continue
        if "es" in value and "url_es" in value:
            objects.append(value)
    seen: set[tuple[int, int]] = set()
    for item in objects:
        label = item["es"].strip().lower()
        url = urljoin("https://www.bcra.gob.ar/", item["url_es"])
        # The live archive has at least one incorrect human-readable label
        # ("Febrero 2011" pointing to temp0212.pdf). Key from the official
        # filename and use the label only when the filename is nonstandard.
        file_match = re.search(r"temp(\d{2})(\d{2}|\d{4})\.pdf", url, re.IGNORECASE)
        if file_match:
            month = int(file_match.group(1))
            year_token = file_match.group(2)
            if len(year_token) == 4:
                year = int(year_token)
            else:
                two_digit_year = int(year_token)
                year = (
                    1900 + two_digit_year
                    if two_digit_year >= 90
                    else 2000 + two_digit_year
                )
        else:
            match = re.match(r"([a-záéíóú]+)\s+(\d{4})", label)
            if not match:
                continue
            month_name = (
                match.group(1)
                .replace("á", "a")
                .replace("é", "e")
                .replace("í", "i")
                .replace("ó", "o")
                .replace("ú", "u")
            )
            month = MONTHS_ES.get(month_name)
            year = int(match.group(2))
        if month is None or (year, month) in seen:
            continue
        if (year, month) < (1999, 12) or (year, month) > (2024, 7):
            continue
        seen.add((year, month))
        source_note = (
            "Spanish IMF reserves/foreign-currency-liquidity template. "
            f"Archive label={item['es']!r}; date keyed from official filename when possible."
        )
        if (year, month) == (2000, 5):
            # The current BCRA archive's temp0500 link serves a document headed
            # May 2001. Preserve that bad response as evidence, then retrieve
            # the original May-2000 BCRA publication from an immutable Internet
            # Archive capture. The archived document remains an official BCRA
            # source; only its present-day carrier is secondary.
            fetcher.get(
                dataset_id="bcra_nedd_wrong_link_anomaly",
                request_id="nedd_2000_05_live_wrong_month",
                institution="BCRA",
                url=url,
                relative_path="nedd/anomalies/2000-05_live_link_wrong_month.pdf",
                vintage_policy="live BCRA archive anomaly retained verbatim",
                observation_start="2000-05-01",
                observation_end="2000-05-01",
                notes=(
                    "Current official archive target is not May 2000; its "
                    "embedded header is retained so the mismatch is auditable."
                ),
                timeout=90,
            )
            url = (
                "https://web.archive.org/web/20000916005139id_/"
                "http://www.bcra.gov.ar:80/pdfs/contad/itemp0500.PDF"
            )
            source_note = (
                "Original BCRA May-2000 NEDD publication recovered through "
                "the Internet Archive capture dated 2000-09-16. Official "
                "document, secondary archival carrier."
            )
        try:
            content = fetcher.get(
                dataset_id="bcra_nedd_pdf",
                request_id=f"nedd_{year:04d}_{month:02d}",
                institution=(
                    "BCRA via Internet Archive"
                    if (year, month) == (2000, 5)
                    else "BCRA"
                ),
                url=url,
                relative_path=f"nedd/{year:04d}-{month:02d}.pdf",
                vintage_policy=(
                    "archived original BCRA publication; Internet Archive capture fixed at 2000-09-16"
                    if (year, month) == (2000, 5)
                    else "archived monthly publication; file Last-Modified retained when exposed"
                ),
                observation_start=f"{year:04d}-{month:02d}-01",
                observation_end=f"{year:04d}-{month:02d}-01",
                notes=source_note,
                timeout=90,
            )
        except RuntimeError:
            if (year, month) != (2000, 5):
                raise
            content = fetcher.copy_verified_local_fallback(
                dataset_id="bcra_nedd_pdf",
                request_id="nedd_2000_05",
                institution="BCRA via Internet Archive",
                original_endpoint=url,
                fallback_source=(
                    RAW_ROOT
                    / "20260725_final_v6"
                    / "nedd"
                    / "2000-05.pdf"
                ),
                relative_path="nedd/2000-05.pdf",
                expected_sha256=(
                    "a8b5b65682954f6c4318bb97707c3df2c633fd5c5108b5366cc381f234792c54"
                ),
                vintage_policy=(
                    "archived original BCRA publication; Internet Archive "
                    "capture fixed at 2000-09-16"
                ),
                observation_start="2000-05-01",
                observation_end="2000-05-01",
                notes=source_note,
            )
        if (year, month) == (2000, 5):
            expected_sha256 = (
                "a8b5b65682954f6c4318bb97707c3df2c633fd5c5108b5366cc381f234792c54"
            )
            observed_sha256 = sha256_bytes(content)
            if observed_sha256 != expected_sha256:
                raise RuntimeError(
                    "Archived BCRA May-2000 payload changed: "
                    f"expected {expected_sha256}, observed {observed_sha256}"
                )
    expected = 296  # Dec-1999 through Jul-2024 inclusive.
    if len(seen) != expected:
        missing = []
        y, m = 1999, 12
        while (y, m) <= (2024, 7):
            if (y, m) not in seen:
                missing.append(f"{y:04d}-{m:02d}")
            m += 1
            if m == 13:
                y += 1
                m = 1
        raise RuntimeError(
            f"NEDD archive index yielded {len(seen)} target months, expected {expected}; "
            f"missing={missing[:20]}"
        )


def finalize_snapshot(fetcher: SnapshotFetcher) -> None:
    manifest_path = fetcher.raw_dir / "retrieval_manifest.csv"
    write_csv_rows(manifest_path, fetcher.rows, MANIFEST_FIELDS)
    copied_manifest = METADATA_ROOT / "retrieval_manifest.csv"
    write_csv_rows(copied_manifest, fetcher.rows, MANIFEST_FIELDS)
    pointer = {
        "snapshot_id": fetcher.snapshot_id,
        "created_at_utc": utc_now_iso(),
        "raw_directory": safe_relpath(fetcher.raw_dir),
        "manifest": safe_relpath(manifest_path),
        "request_count": len(fetcher.rows),
    }
    write_json(METADATA_ROOT / "latest_snapshot.json", pointer)
    write_json(fetcher.raw_dir / "snapshot.json", pointer)


def fetch_snapshot(snapshot_id: str | None = None) -> str:
    ensure_output_dirs()
    selected = snapshot_id or default_snapshot_id()
    raw_dir = RAW_ROOT / selected
    if raw_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable raw snapshot: {raw_dir}. "
            "Choose a new snapshot ID or build from the existing one."
        )
    raw_dir.mkdir(parents=True)
    fetcher = SnapshotFetcher(snapshot_id=selected, raw_dir=raw_dir)
    try:
        fetch_static(fetcher)
        fetch_indec(fetcher)
        from pipeline.common import require_fred_key

        fetch_fred(fetcher, require_fred_key())
        fetch_bcra_v4(fetcher)
        fetch_imf_irfcl(fetcher)
        fetch_ambito(fetcher)
        fetch_nedd_pdfs(fetcher)
        finalize_snapshot(fetcher)
    except Exception:
        # Preserve the partial directory for forensic inspection, but do not point
        # latest_snapshot.json at an incomplete run.
        partial = {
            "snapshot_id": selected,
            "failed_at_utc": utc_now_iso(),
            "completed_request_count": len(fetcher.rows),
            "status": "partial_failed_snapshot",
        }
        write_json(raw_dir / "snapshot_failed.json", partial)
        if fetcher.rows:
            write_csv_rows(raw_dir / "retrieval_manifest_partial.csv", fetcher.rows, MANIFEST_FIELDS)
        raise
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an immutable raw-data snapshot.")
    parser.add_argument("--snapshot-id", default=None, help="Optional unique immutable snapshot ID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = fetch_snapshot(args.snapshot_id)
    print(f"Created immutable snapshot {selected}")


if __name__ == "__main__":
    main()
