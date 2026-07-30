# Forecast-information-set audit

## The origin is not identified by the paper

Figure 4 uses contemporaneous official and parallel exchange-rate growth, but
the paper does not fix the forecast date and time. Three exercises must not be
conflated:

1. **Start-of-month forecast for `t`:** no complete monthly observation from
   `t` is available.
2. **Within-month nowcast for `t`:** daily FX may be averaged only through a
   declared cut date.
3. **End-of-month nowcast for `t`:** complete `t` FX is available, but `t`
   inflation is not yet published.

The delivered underlying dataset does not choose among them. It exposes
availability fields so a later modelling notebook can impose a start-of-month,
day-15, or end-of-month origin without changing the economic series.

## Figure 4 availability

| Predictor | Typical public availability | Day 1 of `t` | Day 15 of `t` | Principal risk |
|---|---|:---:|:---:|---|
| Argentine inflation `t-1`, `t-2` | `t-1` usually around day 12–15 | no | often | intervention-period public reconstruction has no real-time vintage archive |
| effective rate `t-1` | daily/monthly BCRA shortly after month-end | generally | yes | retrospective history and conversion rule |
| M2 `t-1` | short BCRA lag | generally | yes | current historical file may contain revisions |
| registered wages `t-1` | RIPTE up to 45 days after reference month | no | generally no | direct look-ahead if latest `t-1` is used |
| parallel/CCL `t-1` | daily quote | yes | yes | journal endpoint can backfill history |
| parallel/CCL `t` | daily quote | no for full month | partial only | full-month mean is look-ahead at a within-month origin |
| official FX `t-1` | daily/short lag | yes | yes | low, subject to chosen monthly convention |
| official FX `t` | daily | no for full month | partial only | full-month mean is look-ahead |
| activity 3M mean `t-2` | EMAE roughly 48–60 days after reference month | not reliably | not reliably | actual release calendar required; May 2024 was released July 18 |
| real NIR `t-1` | public NEDD normally around day 20–23 of following month | no | generally no | BBVA could have had an internal estimate earlier |
| FX gap `t-1` | derived from two FX levels | yes if both inputs exist | yes | inherits the regime splice |
| US CPI `t-1` | BLS around day 10–14 | no | usually | release-time and revision check |
| Brent/wheat `t-1` | daily markets or later monthly benchmark | partial | often | Pink Sheet monthly vintage may be later/revised |

The 2024 INDEC calendar illustrates why “activity `t-2`” is not automatically
safe: April EMAE was released June 28 and May EMAE July 18
([INDEC technical reports](https://www.indec.gob.ar/indec/web/Institucional-Indec-InformesTecnicos-47)).
BLS released June-2024 CPI on July 11 and July CPI on August 14
([BLS 2024 calendar](https://www.bls.gov/schedule/2024/home.htm)).

RIPTE is particularly problematic. The official rule permits publication up
to 45 days after the reference month, and Resolution 2/2018 states that RIPTE
had been calculated/published only since 2006. Thus 1994–2005 is
retrospective, not a historical real-time predictor
([Resolution 2/2018](https://www.argentina.gob.ar/normativa/nacional/resoluci%C3%B3n-2-2018-306850/texto)).

## Fields delivered

`forte_13_public_proxy_long_with_availability.csv` contains one row for every
month-variable pair and includes:

- reference `date`, variable ID/name, value and unit;
- source, classification and confidence;
- transformation, revision and vintage status;
- a labelled publication-lag rule;
- `estimated_availability_date` and an explicit status saying it is a
  rule-of-thumb rather than an actual release;
- `source_release_date`, blank when not verified;
- `source_last_modified_date`, kept separate from release date;
- `known_by_end_of_month_t` and `known_by_day15_t_plus_1`;
- construction-quality flags, including the NIR blank/absent-cell assumption;
- raw snapshot and retrieval timestamp; and
- links to the raw retrieval manifest and processed
  `audits/variable_raw_source_map.csv`.

The variable-source map expands each underlying variable to the exact raw
request IDs, local paths, retrieval timestamps, source timing fields and
SHA-256 hashes. Hashes are not redundantly copied into every monthly long
row.

For pre-1993 U03, no artificial 55-day EMAE date is assigned: the estimated
date is blank, observed proxy values are labelled
`historical_component_release_dates_unavailable`, and genuinely missing
months remain `not_available`.

## Interpretation

- `yes` and `no` use a documented rule-of-thumb lag, not an asserted
  historical release date.
- `not_vintage_safe` means the value is a later retrospective construction.
- `not_reconstructable` means the public current history cannot establish what
  was known then.
- A raw HTTP `Last-Modified` header is never interpreted as the economic
  release date.
- A complete latest-data row is not evidence that it was available at the
  original forecast origin.

## Required controls before fitting any model

1. Declare one exact origin timestamp for each target month.
2. Enforce `actual_release_timestamp <= origin` when an actual timestamp is
   available.
3. Truncate contemporaneous FX means at the origin; never use a full-month
   mean at day 15.
4. Run a strict specification without contemporaneous `t` FX.
5. Lag RIPTE further when `t-1` was not released.
6. Use actual EMAE release dates; do not assume `t-2` is always available.
7. Compare transition-month parallel-FX growth with the same-source
   sensitivity.
8. Exclude NIR from the primary benchmark. Any later NIR sensitivity must
   report both the baseline and `NA`-conservative constructions.
9. Maintain an information-violation report by origin.

No forecasting features or fitted models are produced in the current phase.
