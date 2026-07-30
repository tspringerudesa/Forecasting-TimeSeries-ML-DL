"""Verify the Data912 MEP/CCL endpoints and their response shape.

Only field names, HTTP status, and any timestamp fields are printed. Prices
are intentionally omitted because the audit concerns source capabilities.
"""

from __future__ import annotations

import json

import requests


BASE_URL = "https://data912.com"


def describe(value: object) -> object:
    if isinstance(value, dict):
        return {
            "fields": sorted(value),
            "timestamps": {
                key: item
                for key, item in value.items()
                if any(token in key.lower() for token in ("date", "time", "fecha"))
            },
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "first_item": describe(value[0]) if value else None,
        }
    return {"type": type(value).__name__}


def main() -> None:
    for endpoint in ("/live/mep", "/live/ccl"):
        response = requests.get(BASE_URL + endpoint, timeout=30)
        payload = response.json() if response.ok else None
        print(
            json.dumps(
                {
                    "endpoint": endpoint,
                    "status": response.status_code,
                    "shape": describe(payload),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
