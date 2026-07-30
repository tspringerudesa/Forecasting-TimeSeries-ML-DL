"""Probe tiny date windows of Ámbito's undocumented blue/CCL endpoints.

These endpoints are unofficial and can change without notice. The script
requests only short windows to establish whether observations exist; it does
not download full histories.
"""

from __future__ import annotations

import json

import requests


PROBES = {
    "blue_2011": (
        "https://mercados.ambito.com/dolar/informal/historico-general/"
        "2011-10-01/2011-10-10"
    ),
    "ccl_2012_absence_check": (
        "https://mercados.ambito.com/dolarrava/cl/historico-general/"
        "2012-01-01/2012-01-10"
    ),
    "ccl_2013": (
        "https://mercados.ambito.com/dolarrava/cl/historico-general/"
        "2013-01-01/2013-01-10"
    ),
    "ccl_2024": (
        "https://mercados.ambito.com/dolarrava/cl/historico-general/"
        "2024-07-01/2024-07-10"
    ),
}


def main() -> None:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/126 Safari/537.36"
            ),
            "Referer": "https://www.ambito.com/",
        }
    )
    for name, url in PROBES.items():
        response = session.get(url, timeout=30)
        if not response.ok:
            print(
                json.dumps(
                    {
                        "probe": name,
                        "status": response.status_code,
                        "observation_count": None,
                        "note": "Endpoint may apply transient anti-bot controls.",
                    },
                    ensure_ascii=False,
                )
            )
            continue
        rows = response.json()
        observations = rows[1:] if rows else []
        print(
            json.dumps(
                {
                    "probe": name,
                    "status": response.status_code,
                    "header": rows[0] if rows else None,
                    "observation_count": len(observations),
                    "first_date_returned": observations[0][0]
                    if observations
                    else None,
                    "last_date_returned": observations[-1][0]
                    if observations
                    else None,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
