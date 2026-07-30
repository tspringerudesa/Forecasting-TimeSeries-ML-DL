"""Small NEDD parser regression probe.

Usage from the replication root:
    python research_probes/probe_nedd_parser.py PATH_WITH_SAMPLE_PDFS

Expected filenames are YYYY-MM.pdf. This script does not write data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_dataset import (
    _nedd_section,
    _parse_nedd_gross,
    _parse_nedd_outflows,
    _parse_nedd_pdf,
    _run_pdftotext,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    for path in sorted(args.directory.glob("????-??.pdf")):
        if args.strict:
            try:
                parsed = _parse_nedd_pdf(path, pd.Timestamp(path.stem + "-01"))
                print(path.stem, parsed)
            except Exception as exc:
                print(path.stem, {"error": repr(exc)})
            continue
        lines = _run_pdftotext(path).splitlines()
        gross, gross_method = _parse_nedd_gross(lines)
        section, total_x, next_total_x = _nedd_section(lines)
        reference_date = pd.Timestamp(path.stem + "-01")
        outflows, methods = _parse_nedd_outflows(
            section, total_x, next_total_x, reference_date
        )
        print(
            path.stem,
            {
                "gross": gross,
                "gross_method": gross_method,
                **outflows,
                "nir": gross + sum(outflows.values()),
                "methods": methods,
                "ma_total_x": total_x,
                "government_total_x": next_total_x,
            },
        )


if __name__ == "__main__":
    main()
