"""Read-only discovery probe for the official Argentine time-series API.

Only the first few search results and one endpoint observation are requested.
Series IDs printed by this script must still be judged on their metadata and
definition; a text match is not an endorsement.
"""

from __future__ import annotations

import argparse
import json

import requests


SEARCH_URL = "https://apis.datos.gob.ar/series/api/search"
SERIES_URL = "https://apis.datos.gob.ar/series/api/series"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = "forte-2024-replication-source-audit/0.1"

    for query in args.queries:
        response = session.get(
            SEARCH_URL,
            params={"q": query, "limit": min(args.limit, 20)},
            timeout=30,
        )
        response.raise_for_status()
        matches = response.json().get("data", [])
        print(json.dumps({"query": query, "match_count_returned": len(matches)}))
        for match in matches:
            field = match.get("field", {})
            series_id = field.get("id")
            sample = None
            if series_id:
                sample_response = session.get(
                    SERIES_URL,
                    params={
                        "ids": series_id,
                        "metadata": "simple",
                        "limit": 1,
                        "sort": "asc",
                    },
                    timeout=30,
                )
                if sample_response.ok:
                    sample = sample_response.json().get("data", [])[:1]
            print(
                json.dumps(
                    {
                        "id": series_id,
                        "description": field.get("description"),
                        "units": field.get("units"),
                        "frequency": field.get("frequency"),
                        "dataset_title": match.get("dataset", {}).get("title"),
                        "source": match.get("dataset", {}).get("source"),
                        "first_sample": sample,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
