# Replication decisions implemented

## Status

The user's instruction to proceed is treated as approval to implement the
closest defensible public construction while keeping unresolved choices
visible. These decisions define the delivered data; they do not claim to
recover Forte's private data exactly.

## Implemented defaults

1. **Panel and frequency.** Monthly calendar, 1965-01 through 2024-07.
   Stocks use end of month; daily market prices/rates use monthly arithmetic
   means unless the institutional source already publishes a monthly average.
2. **Growth convention.** Ordinary simple monthly percentage change,
   `100*(x_t/x_{t-1}-1)`. Log changes remain in the construction-detail file
   as sensitivities.
3. **CPI.** Historical linked INDEC CPI through 2006-12; a three-province
   January-2007 bridge; CIFRA IPC Provincias rates from 2007-02 through
   2016-12; national INDEC CPI rates thereafter. No endpoint rescaling.
4. **Wages.** Official linked non-qualified basic-agreement index through
   1987; official non-qualified ISBIC through 1994-07; RIPTE ratio-linked in
   1994-07 with its growth used from 1994-08.
5. **Activity.** Matched-unit BCRA codes 23+25 (private plus public
   nonfinancial peso credit) deflated by the linked CPI, automobiles plus
   `resto`, and crude steel. PC1 is estimated from z-scored log levels over
   1993-01–2013-12, oriented and mapped to original NSA EMAE, and anchored in
   January 1993. Official printed observations repair 1991–1992; 1989–1990
   vehicles are not interpolated.
6. **Interest.** BCRA code 30 through 2015-11 and BCRA v4 ID 160 thereafter.
   Monthly conversion is the simple nominal convention `TNA*30/365`;
   annual-equivalent compounding is a sensitivity.
7. **Money.** BCRA code 15 for monetary base and code 3543 for M2 total,
   end-of-month growth. Both are underlying Table 1 variables; monetary base
   is not silently added to the Figure 4 final feature list.
8. **Commodities and US CPI.** World Bank Wheat US HRW and Brent long-history
   columns, with FRED EIA/IMF controls; BLS CPI-U NSA via FRED. Early
   backcasted Brent is explicitly weak.
9. **Official FX.** OECD official monthly average via FRED, checked against
   direct BCRA historical and A3500 observations.
10. **Parallel FX.** Regime-specific official, BCRA financial/free, Ámbito
    blue and Ámbito/Rava CCL monthly means. Source transitions, partial months
    and the unified-regime imposed zero gap are flagged. A same-source growth
    sensitivity suppresses transition-month changes.
11. **Net reserves.** Monetary-authority NEDD official assets plus selected
    negative gross outflow details due within twelve months, then CPI-U
    deflation to July-2024 dollars. The baseline treats blank/absent liability
    cells as nil with explicit flags; an `NA`-conservative variant suppresses
    those months.
12. **Vintages.** The delivered panel is latest-data/retrospective. It does
    not pretend to be a first-release panel. Actual release dates remain blank
    when unverified; rule-of-thumb availability dates are stored separately.
13. **Immutability.** Raw and processed snapshot directories cannot be
    overwritten. Every raw request and processed artifact has a SHA-256
    manifest; a variable-to-raw-request map provides the hash lineage.

## Forecasting profiles retained

| Profile | Included variables | Intended use |
|---|---|---|
| `core8` | excludes activity, interest, parallel FX, gap and NIR | longest continuous multivariate starting point |
| `complete_no_nir` | all underlying variables except U13 | broader but shorter complete-panel benchmark without the least defensible NIR convention |
| `full_public_native` | all 13 public proxies, including NIR, with native missing values and quality indicators retained | long-history missing-value experiment for XGBoost Random Forest and LightGBM |

The Core 9 and Core 10 convenience matrices are not used by the forecasting
design and are no longer retained. The comparison intentionally studies the
trade-off between history length and predictor breadth; it is not a
common-sample estimate of the isolated effect of adding predictors.

## Unresolved choices for an exact replication

| Topic | Implemented public choice | What remains unresolved |
|---|---|---|
| CPI 2007–2015 | CIFRA rate chain | author's provincial list, weights, Congreso values, CABA entry, averaging and vintages |
| Wage category | non-qualified | qualified, non-qualified or another combined ISBIC concept |
| Wage splice | RIPTE anchor at 1994-07 | author's exact switch month, source edition and vintage |
| Activity credit | BCRA codes 23+25 / linked CPI; code 23 alone retained as sensitivity | exact “total” credit perimeter, deflator and stock/price timing convention |
| PCA | z-scored log levels, 1993–2013 fit | transformation, loadings window, EMAE variant, mapping and source vintage |
| Parallel FX | documented regime map and monthly mean | exact Alphacast/FIEL/Ámbito keys, quote side, market dates and month-end sensitivity |
| NIR | strict gross-outflow NEDD | BBVA accounting perimeter, swaps/deposits/IMF treatment and pre-1999 history |
| Forecast origin | not imposed in the underlying dataset | exact day/time and first-release policy used by Forte |

## Acceptance boundary

The 13-variable public-proxy dataset is ready for exploratory use and later
feature engineering. The forecasting notebook declares its origin convention
and builds strict lagged features from the retained wide matrices and
construction-quality fields. The long availability file remains the audit
source for any later vintage-safe reconstruction.
