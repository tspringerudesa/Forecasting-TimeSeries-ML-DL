# Forte (2024) public-data replication

This directory contains the critical source audit, immutable retrieval
pipeline and validated 13-variable monthly public-proxy dataset for Federico
D. Forte (2024), *Pronóstico de inflación de corto plazo en Argentina con
modelos Random Forest*, RedNIE Working Paper 340.

## Feasibility verdict

An exact public replication is not feasible because the paper does not expose
its intervention-period CPI recipe, ISBIC category/splice, activity-PCA
parameters, historical parallel-FX source map or BBVA net-reserve workbook.
An economically close and fully traceable public replication is feasible from
1965-01 through 2024-07, with missing observations preserved.

The delivered wide file contains the 13 **underlying Table 1 variables**.
It does not create the 15 lagged/contemporaneous Figure 4 features and does
not fit any forecasting model.

## Authoritative pointers

- `metadata/latest_snapshot.json` — selected retrieval record. The bulky raw
  payloads are reproducible local cache files and are not retained in Git.
- `metadata/latest_build.json` — matching processed snapshot and main files.
- `metadata/latest_validation.json` — validation status and hashes.
- `metadata/file_inventory.csv` — exhaustive path inventory for this
  replication tree, with hashes and final/superseded classifications.
- `metadata/retrieval_manifest.csv` — retained request, source, timestamp and
  SHA-256 manifest for the selected retrieval.

The authoritative build is `20260725_final_v10`. It contains 715 months and
13 columns. Validation passed 34 hard checks out of 34; one additional
warning records the activity proxy's weak 2014–2015 holdout correlation.
The wide-file SHA-256 is
`15b47223c4b626b22e133c1786ca2f924b2f440e2a15c24a77ef44b762019374`.

Superseded raw and processed snapshots are intentionally excluded from the
repository. Their useful conclusions are consolidated in the audit documents.
Use the `latest_*` pointers rather than directory-name sorting.

## Retained benchmark inputs

Inside `data/processed/snapshots/<snapshot_id>/`:

- `forte_13_public_proxy_monthly_1965_2024.csv` — ready-to-use 715-row,
  13-variable wide panel with honest `NA` values.
- `forte_13_construction_details_monthly.csv` — levels, controls,
  sensitivities, splice sources and quality flags.
- `forte_13_public_proxy_long_with_availability.csv` — one
  month-variable row with timing/vintage fields; retained for provenance and
  information-set auditing, not as a modelling matrix.
- `forte_core8_continuous_complete_monthly.csv` — longest continuous
  multivariate subset.
- `forte_full13_complete_monthly.csv` — later complete panel; the forecasting
  notebook drops U13 before feature construction.
- `transformation_manifest.csv` — implemented source and transformation for
  every variable.
- `audits/` — build-immutable wage, activity, FX, NEDD and raw-hash lineage
  tables.
- `processed_file_manifest.csv` — output rows, columns and SHA-256 hashes.
- `dataset_validation_checks.csv`, `coverage_validation.csv` and
  `validation_summary.json` — validation results.

The forecasting benchmark retains three profiles: the complete 714-month
Core 8 history, the 296-month complete later panel after dropping U13, and the
long public panel with native missing values. This directly compares longer
history against broader predictor coverage while preserving a separate
native-missing tree exercise. The generated Core 9 and Core 10 convenience
exports were removed because no retained experiment consumes them.

## Reproduce

Run from this directory in the active `timeseriesforecasting` environment.
`FRED_API_KEY` must already be present in the environment; it is redacted from
manifests and never written to disk.

```powershell
conda run --no-capture-output -n timeseriesforecasting python -m pipeline.run_pipeline --snapshot-id YYYYMMDD_unique
```

The command retrieves a new immutable raw snapshot into the Git-ignored local
cache, builds a matching write-once processed data snapshot and runs the
validation gate. Existing raw and constructed-data snapshot IDs are never
overwritten. New processed snapshots are ignored until deliberately selected
for version control. Re-running the validator refreshes only its three reports
and replaces, rather than duplicates, their processed-manifest rows.

Because the selected raw payload cache is not committed, a fresh retrieval ID
is required to reproduce or revise the build. To validate the retained current
build:

```powershell
conda run --no-capture-output -n timeseriesforecasting python -m pipeline.validate_dataset
```

## Audit documentation

- [`docs/source_audit.md`](docs/source_audit.md) — all 13 mappings.
- [`docs/difficult_variables.md`](docs/difficult_variables.md) — corrected
  working notes for the five hard variables.
- [`docs/difficult_variables_final.md`](docs/difficult_variables_final.md) —
  independent critical audit and final decision table.
- [`docs/forecast_information_set.md`](docs/forecast_information_set.md) —
  look-ahead and release-timing audit.
- [`docs/replication_decisions.md`](docs/replication_decisions.md) —
  implemented defaults and unresolved exact-replication choices.
- [`docs/skill_gap_audit.md`](docs/skill_gap_audit.md) — use of the four
  installed skills and additional-skill decision.
- [`metadata/source_candidates.csv`](metadata/source_candidates.csv) —
  candidate-level IDs, endpoints, units, coverage and rejection reasons.

## Most important cautions

- CPI 2007–2016 is a retrospective institutional reconstruction.
- The historical wage series is machine-extracted from an official PDF and
  has not been independently double-entered.
- Activity has no fabricated 1989–1990 vehicle observations; its in-sample
  level correlation is high but 2014–2015 holdout performance is weak.
- Parallel FX is a low-confidence, researcher-imposed regime splice.
- The continuous NIR baseline assumes blank/absent template liability cells
  are nil and flags all such months; use the `NA`-conservative detail variant
  when that convention is unacceptable.
- Availability dates without direct release evidence are labelled estimates,
  not historical facts.
