# Research probes

Small read-only scripts used to verify identifiers, metadata, endpoint
semantics and parser behavior before full retrieval. They do not fit models.
The production retrieval/build/validation entry points are in `pipeline/`.

## Scripts

- `probe_api_metadata.py` — BCRA v4, Datos Argentina and FRED metadata plus
  small boundary fetches; reads `FRED_API_KEY` from the environment and never
  persists it.
- `probe_indec_search.py` — text discovery in the official time-series API;
  a search hit is not accepted until its ID is checked by metadata/data.
- `probe_bcra_historical_catalog.py` — filters the official BCRA series
  catalogue by definition and code.
- `probe_bcra_historical_coverage.py` — reports first/last/count for selected
  historical BCRA TXT codes.
- `probe_data912_fx.py` — confirms that installed Data912 MEP/CCL capability
  is live rather than long historical data.
- `probe_ambito_fx.py` — checks blue/CCL endpoint dates and schema.
- `probe_cifra_ipc.py` — inspects the small CIFRA workbook's sheet, period and
  boundary rows.
- `probe_nedd_parser.py` — runs the NEDD parser against `YYYY-MM.pdf` samples,
  including embedded-month and monetary-authority-column checks. It is
  working-directory independent.
- `probe_activity_credit_variants.py` — compares BCRA code 23, the
  matched-unit code 23+25 sum, and rejected mixed-currency code 22 using the
  same frozen PCA/mapping diagnostics.

## Examples

```powershell
conda run -n timeseriesforecasting python research_probes/probe_api_metadata.py --provider indec
conda run -n timeseriesforecasting python research_probes/probe_api_metadata.py --provider bcra
conda run -n timeseriesforecasting python research_probes/probe_api_metadata.py --provider fred
conda run -n timeseriesforecasting python research_probes/probe_bcra_historical_catalog.py --term M2
conda run -n timeseriesforecasting python research_probes/probe_bcra_historical_coverage.py
conda run -n timeseriesforecasting python research_probes/probe_data912_fx.py
conda run -n timeseriesforecasting python research_probes/probe_ambito_fx.py
conda run -n timeseriesforecasting python research_probes/probe_cifra_ipc.py
conda run -n timeseriesforecasting python research_probes/probe_nedd_parser.py --strict data/raw/snapshots/SNAPSHOT_ID/nedd
conda run -n timeseriesforecasting python research_probes/probe_activity_credit_variants.py --raw-snapshot data/raw/snapshots/SNAPSHOT_ID
```

## Findings that changed the construction

- The historical wage volume supplies monthly data; the earlier
  “pre-1988 unavailable” conclusion was rejected.
- Automobile IDs have 48 null months in 1989–1992 and crude steel has 24
  null months in 1991–1992 despite much wider outer metadata ranges.
- Official printed tables repair 1991–1992 but not 1989–1990 vehicles.
- Total vehicles are automobiles plus `resto`, not plus `utilitarios`.
- Codes 23+25 are the closest public matched-unit reading of “total peso
  credit”; code 23 alone is narrower, while code 22 mixes currencies and
  begins only in 1990.
- BCRA financial/free FX IDs consist of separate non-null regime blocks and
  contain denomination changes.
- Ámbito blue begins 2002-01-11 and CCL 2013-01-02 in the retrieved endpoint;
  availability does not imply economic relevance in every regime.
- The current BCRA NEDD archive has wrong links for 2000-05 and 2011-05.
  A pinned original BCRA archive capture and official IMF IRFCL data recover
  both months.
- NEDD templates contain two verified omitted-minus anomalies and widespread
  blank/absent liability cells; both are explicitly flagged.
