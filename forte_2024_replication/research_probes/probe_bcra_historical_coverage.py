"""Verify first/last observations for selected BCRA historical TXT series.

The files are streamed and discarded. Only the requested codes' first/last
records, record counts, and response byte counts are printed.
"""

from __future__ import annotations

import json

import requests


FILES = {
    "panser.txt": {"4", "15", "22", "23", "25", "30", "3543", "3544"},
    "din1_ser.txt": {"151"},
}
BASE_URL = (
    "https://www.bcra.gob.ar/archivos/Pdfs/PublicacionesEstadisticas/"
)


def main() -> None:
    for filename, codes in FILES.items():
        first: dict[str, str] = {}
        last: dict[str, str] = {}
        counts = dict.fromkeys(codes, 0)
        byte_count = 0
        with requests.get(BASE_URL + filename, stream=True, timeout=90) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                byte_count += len(raw_line) + 1
                line = raw_line.decode("latin-1", errors="replace")
                code = line.partition(";")[0].strip()
                if code in codes:
                    first.setdefault(code, line)
                    last[code] = line
                    counts[code] += 1
        print(
            json.dumps(
                {
                    "file": filename,
                    "response_bytes": byte_count,
                    "series": {
                        code: {
                            "count": counts[code],
                            "first": first.get(code),
                            "last": last.get(code),
                        }
                        for code in sorted(codes, key=int)
                    },
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
