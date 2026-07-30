from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.common import METADATA_ROOT, PROJECT_ROOT, sha256_file, write_csv_rows


DEFAULT_SELECTED_SNAPSHOT = "20260725_final_v10"
INVENTORY_PATH = METADATA_ROOT / "file_inventory.csv"


def classify(path: Path, selected_snapshot: str) -> tuple[str, bool]:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    if relative == ".gitignore":
        return "repository_configuration", True
    if relative == "README.md":
        return "project_documentation", True
    if relative.startswith("docs/"):
        return "audit_documentation", True
    if relative.startswith("metadata/"):
        return "metadata", True
    if relative.startswith("pipeline/"):
        return "pipeline_code", True
    if relative.startswith("research_probes/"):
        return "validation_probe", True
    if relative.startswith("data/raw/snapshots/"):
        return "regenerable_raw_cache", False
    if relative.startswith(f"data/processed/snapshots/{selected_snapshot}/"):
        return "selected_processed_snapshot", True
    if relative.startswith("data/processed/snapshots/"):
        return "superseded_processed_snapshot", False
    if relative.startswith("data/interim/"):
        return "working_audit_or_log", False
    return "other", False


def write_inventory(selected_snapshot: str) -> Path:
    rows: list[dict[str, object]] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.resolve() == INVENTORY_PATH.resolve():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        category, selected_final = classify(path, selected_snapshot)
        rows.append(
            {
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "category": category,
                "selected_final": selected_final,
                "inventory_status": "hashed",
            }
        )

    # A manifest cannot contain a stable hash of itself. Retain a visible
    # self-row so the inventory is also an exhaustive path listing.
    rows.append(
        {
            "relative_path": INVENTORY_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": "",
            "sha256": "",
            "category": "metadata",
            "selected_final": True,
            "inventory_status": "self_manifest_not_hashed",
        }
    )
    rows.sort(key=lambda row: str(row["relative_path"]))
    write_csv_rows(
        INVENTORY_PATH,
        rows,
        [
            "relative_path",
            "bytes",
            "sha256",
            "category",
            "selected_final",
            "inventory_status",
        ],
    )
    return INVENTORY_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hash and classify every file in the replication directory."
    )
    parser.add_argument(
        "--selected-snapshot",
        default=DEFAULT_SELECTED_SNAPSHOT,
        help="Snapshot ID to mark as the authoritative raw and processed build.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_inventory(args.selected_snapshot)
    print(f"Wrote {output.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
