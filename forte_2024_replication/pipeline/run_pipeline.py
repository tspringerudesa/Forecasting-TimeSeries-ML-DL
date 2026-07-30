from __future__ import annotations

import argparse

from pipeline.build_dataset import build_dataset
from pipeline.fetch_raw import fetch_snapshot
from pipeline.validate_dataset import validate_latest_build


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch, construct and validate the Forte public-proxy dataset."
    )
    parser.add_argument("--snapshot-id", default=None, help="Unique raw/build snapshot ID.")
    parser.add_argument(
        "--reuse-latest",
        action="store_true",
        help="Skip retrieval and rebuild from metadata/latest_snapshot.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse_latest:
        selected = None
    else:
        selected = fetch_snapshot(args.snapshot_id)
    build = build_dataset(selected)
    validation = validate_latest_build(build.snapshot_id)
    print(
        f"Ready: snapshot={build.snapshot_id}, months={len(build.wide)}, "
        f"variables={len(build.wide.columns)}, validation={validation['status']}."
    )


if __name__ == "__main__":
    main()
