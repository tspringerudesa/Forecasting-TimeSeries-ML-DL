# Source and construction audit

## Verdict

A public, economically close replication is feasible, but an exact replication
is not. The delivered panel runs from January 1965 through July 2024 and keeps
unresolved observations as `NA`. It does not interpolate, extrapolate, or
silently treat a retrospective series as real-time data.

The strongest long-history variables are monetary base, M2, US CPI, and the
official exchange rate. The five difficult variables remain proxies:

- 2007–2016 Argentine CPI uses a transparent institutional reconstruction;
- wages splice a historical collective-agreement index to RIPTE;
- pre-1993 activity is a retrospective PCA proxy with a documented 1989–1990
  automobile-data gap;
- the parallel rate is explicitly regime dependent;
- net reserves use the public IMF/BCRA reserves-liquidity template rather than
  BBVA Research's unpublished accounting workbook.

## Thirteen variables versus fifteen model features

Table 1 contains **13 underlying economic variables** (`U01`–`U13`). Figure 4
contains **15 model features** because inflation has two lags and official and
parallel exchange-rate growth each appear both contemporaneously and lagged.
Monetary base is a Table 1 alternative to M2; it is not in the final Figure 4
configuration merely because it was compared.

No lagged feature matrix and no forecasting model is built in this phase.

## Construction adopted

| ID | Underlying variable | Public construction in the delivered panel | Usable coverage in target window | Status | Confidence |
|---|---|---|---|---|---|
| U01 | Argentine CPI inflation | INDEC historical linked CPI through 2006-12; three-province bridge in 2007-01; CIFRA IPC Provincias rates through 2016-12; national INDEC CPI thereafter | 1965-01–2024-07 | close proxy | medium |
| U02 | Registered nominal wages | official linked non-qualified basic-agreement wage index through 1987; ISBIC 1988–1994-07; ratio-linked RIPTE growth from 1994-08 | growth from 1965-02 | close proxy | medium |
| U03 | Monthly activity | PC1 of z-scored log levels of CPI-deflated BCRA codes 23+25 (private plus public nonfinancial peso credit), automobiles plus `resto`, and crude steel; 1993-01–2013-12 loadings/log-EMAE mapping; original NSA EMAE thereafter; trailing three-month simple growth | 1965-04–1988-12 and 1991-04–2024-07; `NA` in 1965-01–03 and 1989-01–1991-03 | close proxy | low |
| U04 | Nominal effective monthly rate | BCRA code 30 through 2015-11, policy rate ID 160 thereafter; `TNA*30/365` | 1965–2024, except 1990-01–03 source gap | close proxy | medium |
| U05 | Monetary-base growth | BCRA code 15, end-of-month stock | 1965-01–2024-07 | exact conceptual match | high |
| U06 | M2-total growth | BCRA code 3543, end-of-month stock | 1965-01–2024-07 | exact conceptual match | high |
| U07 | Wheat-price growth | World Bank Pink Sheet, Wheat US HRW | 1965-01–2024-07 | close proxy to undisclosed Haver series | medium |
| U08 | Brent-price growth | World Bank Pink Sheet Brent column; EIA/FRED control | 1965-01–2024-07 | weak before a comparable Brent market; close thereafter | medium |
| U09 | US CPI inflation | BLS CPI-U NSA via FRED `CPIAUCNS` | 1965-01–2024-07 | exact conceptual match | high |
| U10 | Official ARS/USD growth | OECD official monthly average via FRED `ARGCCUSMA02STM`; direct A3500 overlap control | 1965-01–2024-07 | close proxy | medium |
| U11 | Relevant parallel/free-market rate growth | official rate in unified regimes; normalized BCRA financial/free quotes in verified blocks; blue/CCL only in modern control regimes | 1967-03 onward with explicit early and controlled-regime gaps | weak proxy | low |
| U12 | Exchange-rate gap | `100*(parallel/official-1)` using the same monthly-average convention | inherits U11 | weak proxy | low |
| U13 | Real net international reserves | NEDD official assets minus gross monetary-authority outflows due within 12 months; CPI-U to July-2024 dollars | 1999-12–2024-07 under the flagged blank/absent-as-zero convention; conservative non-imputed variant is sparse | close public proxy | medium |

The 1965–1987 wage levels are machine-extracted from an official publication,
not supplied as an official machine-readable series. The audit retains every
raw token: 265 of 276 monthly cells use deterministic OCR normalization and 11
use manually inspected overrides. It has not undergone independent
double-entry transcription, so extraction confidence is medium.

The primary growth convention is the ordinary monthly percentage change,
`100*(x_t/x_{t-1}-1)`. The paper says monthly variation but does not state a
log transformation. Log-growth alternatives are retained in the construction
detail file rather than silently imposed as the baseline.

For U03, the 1993–2013 PCA/mapping interval is an analyst choice, not a
parameter recovered from Forte. With the semantically preferred total-peso
credit sum, its 0.874 correlation is an in-sample log-level diagnostic;
2014–2015 holdout correlations are only about 0.372 in log levels and 0.392
in monthly log growth. The narrower code-23-only sensitivity raises the
in-sample level correlation to 0.924, illustrating how definition choice can
optimize the headline statistic without identifying the author's input. This
does not validate
short-run stability.

## Verified identifiers and source files

The pipeline freezes metadata and observations before construction. The
principal verified identifiers are:

- CPI: historical level `178.1_NL_GENERAL_0_0_13`; national level
  `148.3_INIVELNAL_DICI_M_26`; January-2007 bridge levels
  `195.1_NIVEL_GENERAL_0_0_13`, `196.1_NIVEL_GENERAL_2014_0_13` and
  `197.1_NIVEL_GENERAL_2014_0_13`; plus the CIFRA IPC Provincias workbook;
- RIPTE: `158.1_REPTE_0_0_5`;
- automobiles: `330.3_PRODUCCIONLES__22` plus
  `330.3_PRODUCCIONSTO__16`;
- crude steel: `359.3_ACERO_CRUDUDO__11`;
- EMAE: `10.3_ISOM_1993_M_29`, `10.3_ISD_1993_M_31`,
  `143.3_NO_PR_2004_A_21`, and `143.3_NO_PR_2004_A_31`;
- historical FX: `175.1_DR_FINANTA_0_0_22` and
  `175.1_DR_LIBRNTA_0_0_17`;
- BCRA historical file codes: private peso credit `23`, public peso credit
  `25`, rejected mixed-currency total credit `22`, interest `30`, monetary
  base `15`, and M2 total `3543`;
- BCRA API v4: gross reserves `1`, passive USD repo `76`, FX current accounts
  `1243`, and policy rate `160`;
- FRED: `CPIAUCNS`, `ARGCCUSMA02STM`, `DCOILBRENTEU`,
  `PWHEAMTUSDM`, and `POILBREUSDM`.

Candidate-level endpoints, units, revisions, publication lags, verified
coverage, and rejection reasons are in
[`source_candidates.csv`](../metadata/source_candidates.csv). The official
Argentina time-series API is documented by
[Datos Argentina](https://www.argentina.gob.ar/datos-abiertos/api-series-de-tiempo);
the historical BCRA files are documented by the
[BCRA series catalogue](https://www.bcra.gob.ar/consulta-de-series-estadisticas-en-formato-txt/).

## Revisions, vintages, and release dates

The raw manifest separates:

- retrieval timestamp;
- HTTP `Last-Modified`;
- economic release date;
- vintage policy;
- requested observation range;
- byte count and SHA-256.

`Last-Modified` is **not** treated as a release date. This matters for the NEDD
archive: old files have server modification dates years after their reference
month. Observation-level output therefore leaves `source_release_date` blank
unless it is actually known and stores an explicitly labelled
rule-of-thumb availability date separately.

The delivered panel is a latest-data/retrospective panel, not a real-time
vintage panel. CIFRA has no contemporaneous workbook-vintage archive.
Resolution 2/2018 states that RIPTE had been calculated and published only
since 2006, so 1994–2005 values are retrospective. The PCA activity proxy,
EMAE and Argentine API histories likewise lack complete first-release
vintages
([Resolution 2/2018](https://www.argentina.gob.ar/normativa/nacional/resoluci%C3%B3n-2-2018-306850/texto)).
See
[`forecast_information_set.md`](forecast_information_set.md) before using the
data in a forecasting experiment.

## Reproducibility and validation

The selected retrieval record and every retained processed output are
identified in `metadata/latest_snapshot.json` and
`metadata/latest_build.json`. The request and hash record is retained in
`metadata/retrieval_manifest.csv`; regenerable raw payloads are excluded from
Git. Retrieval and construction refuse to overwrite an existing local raw or
constructed data directory. Revalidation refreshes only its validation
reports and replaces their manifest rows. During a fresh build, every raw file
is re-hashed and checked against its manifest before parsing.

The validation gate checks:

- the exact 715-month index and 13-column schema;
- duplicate and infinite values;
- CPI, exchange-gap, and reserve-deflation formulas;
- the unfilled 1989–1990 activity source gap;
- nonmissing intervention-period CPI and post-splice wages, plus fixed ISBIC
  anchor checks;
- the July-2024 NEDD reserve arithmetic;
- all 296 NEDD months: 294 valid current-archive BCRA files, the original
  May-2000 BCRA file recovered from a fixed Internet Archive capture, and an
  official IMF IRFCL replacement for the current archive's wrong May-2011
  target;
- the distinction between a valid IMF series with observations and valid
  codelist IDs for which Argentina reports no observations;
- NEDD blank/absent-cell counts and the deliberately sparse conservative
  variant;
- activity API-versus-file equality, pre-patch null blocks, archival-repair
  totals, PCA formula/anchor/loading and independently recomputed in-sample
  and holdout diagnostics;
- parallel-rate carrier transitions, observed block starts, same-source
  growth sensitivity and unified-market zero-gap convention;
- exact IMF codelist membership, valid-empty Argentina detail responses,
  reverse-repo inflow diagnostics, and raw-source hashes for every underlying
  variable.

The final machine-readable reports are in the selected processed snapshot:
`dataset_validation_checks.csv`, `coverage_validation.csv`,
`construction_metrics.json`, and `processed_file_manifest.csv`.
