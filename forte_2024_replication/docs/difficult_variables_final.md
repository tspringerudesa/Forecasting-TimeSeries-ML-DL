# Final critical audit of the five difficult variables

## Scope and audit standard

This is the final source-and-construction decision record for the five
variables that cannot be copied directly from one stable public series. It
separates:

- what Forte (2024) says;
- coverage actually observed in a metadata request, file, or small data fetch;
- coverage stated only in documentation;
- a defensible public construction; and
- information still required for an exact replication.

No missing historical value is interpolated or silently backcast. The
implemented baseline uses ordinary monthly percentage variation,
`100*(x_t/x_{t-1}-1)`. Log changes are retained as diagnostics because the
paper says “monthly variation” but does not specify logarithms.

Confidence means confidence in the **public construction**, not confidence
that it exactly reproduces the author's private dataset.

## 1. CPI reconstruction, 2007–2015

### Paper concept

The target is monthly Argentine CPI inflation. For the intervention period,
the paper describes an average involving provincial indicators, IPC Congreso
and CABA. It does not report the exact provincial set, weights, entry and exit
dates, missing-data rule, or whether the average is applied to index levels or
monthly rates.

### Independently verified sources

| Source | Verified evidence and coverage | Finding |
|---|---|---|
| Historical INDEC CPI-GBA, ID `178.1_NL_GENERAL_0_0_13` | API metadata and observations include the pre-intervention boundary; the pipeline uses it through 2006-12 | suitable official pre-2007 carrier, but GBA rather than a national index |
| CIFRA-CTA `IPC-Provincias-2007-2018.xlsx` | workbook has 144 monthly rows, 2007-01–2018-12, January 2014=100 | strongest public institutional intervention-period reconstruction |
| CIFRA methodology | names a consumption-weighted, changing set of ten provincial/CABA indices, with imputation when sources disappear | it is not a fixed “IPC-9,” and it does not include IPC Congreso |
| Mendoza, ID `195.1_NIVEL_GENERAL_0_0_13` | API observations include 2006-12 and 2007-01, but have a 48-month gap from 2012-04 through 2016-03 | usable for the January-2007 bridge, not for a continuous fixed panel |
| Neuquén, ID `196.1_NIVEL_GENERAL_2014_0_13` | API observations include 2006-12–2016-12 | usable provincial sensitivity component |
| San Luis, ID `197.1_NIVEL_GENERAL_2014_0_13` | API observations include 2006-12–2016-12 | usable provincial sensitivity component |
| CABA, ID `193.1_NIVEL_GENERAL_JULI_0_13` | verified from 2012-07 | cannot represent 2007–2012 |
| National INDEC CPI, ID `148.3_INIVELNAL_DICI_M_26` | level exists at 2016-12 and monthly thereafter | correct post-intervention carrier; its first computable rate is 2017-01 |
| IPC Congreso | press releases establish its historical existence, but no stable primary machine-readable archive, official mnemonic, complete methodology or complete level series was verified | unavailable for a reproducible automated baseline |

CIFRA's own page describes the component provinces, expenditure weights and
2007–2018 period and links both the workbook and methodology
([CIFRA](https://centrocifra.org.ar/estadisticas/ipc-provincias/)). INDEC
separately warns users about the quality of official statistics for
2007–2015, so official intervention-period INDEC inflation cannot be relabelled
as an exact substitute
([INDEC archive notice](https://www.indec.gob.ar/indec/web/Institucional-Indec-InformacionDeArchivo-1)).

### Alternatives compared

1. **CIFRA rate chain — chosen.** Use official historical CPI through 2006-12;
   use the arithmetic mean of the Mendoza, Neuquén and San Luis simple monthly
   rates only for 2007-01; use CIFRA's simple monthly rates from 2007-02
   through 2016-12; then use national INDEC rates from 2017-01.
2. **Fixed provincial panel — sensitivity.** The equal mean of Neuquén and San
   Luis remains continuous. Over the intervention interval its monthly-rate
   correlation with CIFRA is about 0.875 and RMSE is about 0.45 percentage
   point. It sacrifices CIFRA's broader geography but avoids a changing panel.
3. **Mendoza–Neuquén–San Luis fixed panel — rejected as continuous baseline.**
   Mendoza's four-year internal gap would require unreported imputation.
4. **Official INDEC 2007–2015 — rejected as an exact replication.** It
   contradicts the paper's stated reconstruction and INDEC's own archive
   warning.
5. **Endpoint-rescaled chain — rejected.** Forcing the reconstructed index to
   equal December-2016 INDEC would change monthly inflation or inject a
   future endpoint into earlier observations.

### Hidden inconsistencies and final decision

- CIFRA changes composition and imputes components; a fixed-panel sensitivity
  answers a different question.
- The January-2007 rate is not available from the CIFRA level alone because
  its workbook starts in January; the three-province bridge is an explicit
  researcher assumption.
- National CPI contributes rates only from January 2017 even though the
  intervention is usually described as ending in 2015.
- Provincial and CIFRA histories are retrospective and do not supply a
  complete first-release vintage archive.
- Averaging rates, averaging levels and geometric averaging are not
  equivalent.

**Chosen construction:** the CIFRA rate chain above.  
**Confidence:** medium.  
**Needed for exact replication:** the author's component files, exact weights,
entry/exit and missing-data rules, Congreso source, CABA treatment,
level-versus-rate convention, and the vintage used at every forecast origin.

## 2. ISBIC/RIPTE wage splice

### Paper concept

The paper calls for monthly registered nominal wage growth, using ISBIC from
1962 and RIPTE from 1994. Those labels do not by themselves identify an ISBIC
category or an overlap/splice formula.

### Independently verified sources

INDEC's 1991 methodology volume contains monthly linked general indices for
qualified and non-qualified basic-agreement wages through 1987 and explains
their link to the base-1988 successor. It states that the old index was
published monthly and that retroactive collective agreements can revise
earlier values. The source is an official digitized PDF, not a maintained API
series
([INDEC 1991 volume](https://biblioteca.indec.gob.ar/bases/minde/4si9_4.pdf)).

The separate official ISBIC table contains monthly qualified and
non-qualified columns from 1988-01 through 2016-09. RIPTE ID
`158.1_REPTE_0_0_5` was verified at 1994-07, 1994-08 and 2024-07.

Parser assertions protect against the principal OCR/locale failure:

- old linked index, 1987-12: `36.860`;
- modern ISBIC non-qualified, 1988-01: `40.336`;
- 1988-12: `184.600`;
- 1989-01: `198.3`;
- 1994-07: `219773.1`.

Interpreting decimal-comma tokens such as `40,336` as forty thousand rather
than `40.336` is rejected.

### Alternatives compared

1. **Non-qualified linked index + RIPTE — chosen.** Use the official linked
   non-qualified series through 1987-12, modern non-qualified ISBIC through
   1994-07, ratio-link RIPTE at 1994-07 and apply RIPTE growth from 1994-08.
2. **Qualified category — retained as a future sensitivity.** It is equally
   present in the publications, but Forte does not identify which category he
   used.
3. **Average of qualified and non-qualified — not adopted.** No paper wording
   establishes that averaging rule.
4. **Modern general wage index backcast — rejected.** It changes population,
   pay concept and institutional coverage.
5. **Annual/quarterly API series interpolated to months — rejected.** It would
   fabricate within-period wage movements.

### Hidden inconsistencies and final decision

- ISBIC is a basic collective-agreement wage index for manufacturing and
  construction; RIPTE is taxable remuneration of stable registered workers.
  A ratio link removes a level discontinuity, not this conceptual break.
- The old and base-1988 ISBIC differ in geography, agreement coverage,
  weighting, occupations and formulas.
- The historical table is OCR-transcribed. Raw tokens and correction methods
  are retained in
  `data/processed/snapshots/<snapshot_id>/audits/historical_wage_transcription.csv`.
- Some OCR annual-mean tokens are unreliable even when the twelve monthly
  entries pass structural checks. They are not used to alter monthly values.
- Both ISBIC and RIPTE can be revised. RIPTE `t-1` often was not public at a
  mid-month nowcast origin.

**Chosen construction:** non-qualified historical index through 1987, modern
ISBIC through 1994-07, ratio-linked RIPTE thereafter. Monthly growth is
available from 1965-02 through 2024-07.  
**Confidence:** medium.  
**Needed for exact replication:** the author's chosen category, source edition
and transcription, exact switch month and splice rule, and vintage/release
calendar.

## 3. Pre-1993 PCA activity proxy

### Paper concept

Before EMAE begins, the paper uses the first principal component of real total
peso credit, automobile production and steel production, reporting a
post-1993 correlation above 80 percent with EMAE. It does not state the exact
credit perimeter, deflator, standardization window, PCA estimation window,
EMAE seasonal-adjustment variant, sign convention, mapping or vintage.

### Independently verified inputs

| Component | Verified candidate | Coverage result |
|---|---|---|
| Credit | BCRA historical codes `23+25`, peso credit to the private and public nonfinancial sectors, end of month | both monthly from 1940; closest public reading of “total en pesos,” though Forte's exact perimeter is undisclosed |
| Vehicles | ID `330.3_PRODUCCIONLES__22` plus `330.3_PRODUCCIONSTO__16` | monthly from 1963; both components are null for 1989-01–1992-12 and the chosen `resto` component ends in 2015-12 |
| Steel | ID `359.3_ACERO_CRUDUDO__11` | monthly from 1965, but API values are null for 1991-01–1992-12 |
| EMAE | original ID `10.3_ISOM_1993_M_29`; SA control `10.3_ISD_1993_M_31`; later base-2004 IDs | original and SA monthly observations verified from 1993 |

The API's `utilitarios` field is not the residual vehicle category. Total
vehicles are cars plus `resto`. Official Ministry tables recover all twelve
vehicle and steel observations for 1991; revised INDEC tables recover all
twelve for 1992. Their annual sums are 138,958 and 262,022 vehicles, and
2,972.0 and 2,679.9 thousand tonnes of crude steel, respectively. The 1991
monthly steel rows sum to 2,972.0 while the printed annual total is 2,972.3;
the 0.3-thousand-tonne rounding/reconciliation difference is retained rather
than allocated to a month. No
defensible complete official monthly vehicle table was found for 1989–1990.
The build also proves that the API patch cells are still null and that API
values/missingness equal the separately archived official CSVs before
inserting the printed 1991–1992 observations.

Primary files are the
[official automotive CSV](https://infra.datos.gob.ar/catalog/sspm/dataset/330/distribution/330.3/download/datos-historicos-industria-automotriz-unidades-mensuales.csv),
[official steel CSV](https://infra.datos.gob.ar/catalog/sspm/dataset/359/distribution/359.3/download/datos-historicos-de-la-industria-siderurgica-datos-mensulaes.csv),
[1991 Ministry tables](https://cdi.mecon.gov.ar/greenstone/collect/iet/index/assoc/HASH0b19.dir/doc.pdf),
and [revised 1992 INDEC tables](https://biblioteca.indec.gob.ar/bases/minde/epidic93.pdf).

### Methods compared

1. **Standardized log-level PCA — chosen.** Sum matched-unit codes `23+25`,
   deflate by the linked Argentine CPI, take logs of all three components,
   standardize using 1993-01–2013-12, estimate PC1, orient the sign positively
   to original NSA EMAE, regress log EMAE on PC1 over the same overlap, and
   anchor the mapped proxy at January 1993. Use the proxy through 1992 and
   EMAE thereafter.
2. **PCA of monthly log changes — rejected as baseline.** The validation probe
   produced materially weaker agreement with EMAE; it does not reproduce the
   reported greater-than-80-percent level correlation.
3. **Pre-1993-only loadings — retained as sensitivity.** Fitting
   standardization and loadings over 1965–1988 avoids using post-1993
   component values to define the historical factor, but the subsequent EMAE
   level mapping still uses overlap data.
4. **Interpolating 1989–1990 vehicles — rejected.** There is no evidentiary
   basis for allocating annual output across months.
5. **Private-sector code `23` alone — retained as sensitivity.** It is
   narrower than “total peso credit”; it yields a higher in-sample level
   correlation, which underscores the definition-selection risk.
6. **BCRA code `22` — rejected.** Although labelled total credit, it mixes
   domestic- and foreign-currency loans and starts only in 1990.
7. **Cars plus `utilitarios` — rejected.** It omits the verified residual
   commercial-vehicle category.

The chosen log-level PCA has an in-sample overlap correlation of about 0.874
with original EMAE, consistent with the paper's statement. Its in-sample
monthly-growth correlation is only about 0.526, however, and a natural
2014–2015 post-fit holdout falls to about 0.372 in log levels and 0.392 in
monthly log growth. The private-credit-only sensitivity reaches 0.924 in
sample. The high headline correlation is therefore trend-dominated,
definition-sensitive, in-sample evidence rather than validation of short-run
stability.

The panel computes ordinary monthly activity growth and a trailing
three-month mean. Reindexing to a complete monthly calendar occurs before
growth, preventing a false 1988-to-1991 change across the missing block.

### Hidden inconsistencies and final decision

- Summing private and public nonfinancial peso credit and CPI-deflating the
  result are defensible assumptions, not disclosed paper choices.
- PCA on log levels can be driven by common trend; the high level correlation
  is not evidence of equally strong short-run growth tracking.
- The 1993–2013 fit is retrospective and uses future overlap information.
- Original NSA EMAE is internally consistent with NSA physical inputs, but
  the paper does not identify original versus seasonally adjusted EMAE.
- The monthly activity level is unavailable in 1989–1990. Consequently the
  three-month growth measure is also unavailable through 1991-03.

**Chosen construction:** standardized log-level PC1, overlap-mapped to original
EMAE, using BCRA codes `23+25`, with archival 1991–1992 repairs and an explicit
1989–1990 gap. The first three-month-growth observation is 1965-04.  
**Confidence:** low overall.  
**Needed for exact replication:** the author's credit series and deflator,
1989–1990 vehicle observations, exact PCA/standardization window and
loadings, EMAE variant/vintage, and sign/scale/splice code.

## 4. Historical parallel exchange-rate splice

### Paper concept

The paper asks for the economically relevant parallel/free-market exchange
rate in each period and cites Alphacast, FIEL and Ámbito. That is a
regime-dependent concept: blue, free, financial, MEP and CCL are not
interchangeable.

### Independently verified sources and unit breaks

The official historical API IDs are:

- financial sale, `175.1_DR_FINANTA_0_0_22`, with usable blocks
  1971-09-20–1976-03-04, 1981-06-22–1981-12-23 and
  1982-07-06–1983-01-02;
- free sale, `175.1_DR_LIBRNTA_0_0_17`, with usable blocks
  1976-01-08–1981-01-01, 1983-01-03–1987-01-25 and
  1987-11-02–1992-01-01.

Their raw values contain currency-reform scale changes. Normalization to
current ARS applies audited factors of `1e11`, `1e8`, `1e7`, `1e4` or `1e2`
according to the quote date. The daily normalization table is retained in
`data/processed/snapshots/<snapshot_id>/audits/historical_fx_normalization_daily.csv`.

Ámbito's undocumented endpoint returns blue observations from January 2002
and CCL observations from January 2013. Requests use `[start,end)` semantics;
the retrieval therefore asks through the first day of the next year (or
2024-08-01) and records quote counts and first/last dates. Data912's installed
skill returned current MEP/CCL but its historical route was unavailable, so it
cannot support the long splice.

### Alternatives compared and chosen regime map

| Period | Chosen carrier | Reason or caveat |
|---|---|---|
| 1965-01–1967-02 | missing | no verified public series |
| 1967-03–1971-08 | official rate | documented unified market; gap is zero by construction |
| 1971-09–1975-12 | BCRA financial sale | verified active block |
| 1976-01–1980-12 | BCRA free sale | verified active block |
| 1981-01–1981-05 | official rate | unified interval |
| 1981-06–1981-12 | BCRA financial sale | verified active block |
| 1982-01–1982-06 | missing | no verified relevant quote |
| 1982-07–1982-12 | BCRA financial sale | verified active block |
| 1983-01–1987-01 | BCRA free sale | verified active block |
| 1987-02–1987-10 | missing | no verified relevant quote |
| 1987-11–1989-04 | BCRA free sale | verified active block |
| 1989-05–1990-01 | missing | stale and mixed-denomination observations rejected |
| 1990-02–1990-05 | BCRA free sale | normalized, but low confidence |
| 1990-06–2011-09 | official rate | unified market |
| 2011-10–2012-12 | Ámbito blue sale | exchange-control regime; no verified CCL endpoint yet |
| 2013-01–2015-12 | Ámbito/Rava CCL | paper's modern financial-rate concept |
| 2016-01–2019-08 | official rate | unified interval |
| 2019-09–2024-07 | Ámbito/Rava CCL | renewed control regime |

Rejected alternatives are:

- backcasting modern CCL across periods when the instrument was unavailable
  or not the relevant marginal market;
- using Ámbito blue continuously from 2002 merely because the endpoint
  backfills it;
- treating the outer first/last dates of BCRA IDs as continuous coverage;
- using the stale May-1989–December-2011 tail of the free-rate ID;
- using Data912 live snapshots as history; and
- asserting a public FIEL series without a verified public key or download.

### Hidden inconsistencies and final decision

- Monthly means are chosen because the other FX carrier is a monthly average;
  month-end levels would materially change growth and the gap.
- Boundary-month means can mix regimes. Quote count, first date, last date and
  partial-month flags must accompany the value.
- Setting parallel equal to official in unified regimes encodes a zero gap; it
  is a transparent economic convention, not a directly observed parallel
  quote.
- The modern journal endpoints may be backfilled or revised and expose no
  historical release vintage.
- Growth across a market-source boundary combines price change and definition
  change.

The output distinguishes two boundary concepts. The **carrier-transition**
flag marks ten changes in selected source
(1971-09, 1976-01, 1981-01, 1981-06, 1983-01, 1990-06, 2011-10, 2013-01,
2016-01 and 2019-09). The **observed-block-start** flag additionally marks the
first observation and resumptions after explicit gaps (1967-03, 1982-07,
1987-11 and 1990-02). A same-source growth sensitivity is `NA` at carrier
transitions and otherwise equals the baseline; the baseline retains the
economically observed boundary change.

**Chosen construction:** the regime map above, monthly arithmetic means of
sale/reference quotes, with explicit gaps and simple monthly growth.
Achievable coverage starts 1967-03 with gaps; it is uninterrupted from
1990-06 through 2024-07.  
**Confidence:** low.  
**Needed for exact replication:** the author's Alphacast/FIEL/Ámbito extracts,
series keys, regime dates, currency conversions, quote side, monthly
aggregation and boundary treatment.

## 5. Public approximation of net international reserves

### Paper concept

The paper defines BBVA Research net international reserves as gross BCRA
reserves less foreign-currency liabilities due within twelve months, in
constant US dollars. The underlying BBVA accounting workbook is not public.

### Observable, assumed and unavailable items

**Directly observable**

- monetary-authority official reserve assets in Section I.A of the BCRA/IMF
  reserves and foreign-currency liquidity template;
- predetermined principal and interest outflows within one year;
- gross short forward/futures positions;
- negative repo, trade-credit and other-payable outflows;
- BCRA gross reserves ID `1` for a robustness variant; and
- US CPI-U NSA, FRED `CPIAUCNS`, for constant-dollar conversion.

**Accounting assumptions**

- use only the monetary-authority column, not central government;
- sum negative gross outflows and do not net positive inflows;
- treat a genuinely blank or absent liability cell as nil and flag it;
- exclude contingent drains in Section III;
- interpret all selected maturity buckets as due within twelve months; and
- use the reference-month template, even though it is published later.

This blank/absent-as-zero convention affects 264 of 296 months: row 1 is
absent-assumed-zero in 16, row 2 is blank-assumed-zero in 257 and
absent-assumed-zero in seven, and row 3 is absent-assumed-zero in 80.
The build therefore also emits an `NA`-conservative variant that suppresses
NIR whenever any selected category depends on that convention.

**Unavailable for an exact reconstruction**

- BBVA's item mapping, sign rules, overrides and revisions;
- treatment of the China swap, reserve requirements/foreign-currency
  deposits, SEDESA, ALADI, BIS/repos and IMF obligations not cleanly mapped in
  the public template;
- any historical BBVA series before the public template begins; and
- BBVA real-time vintages.

### Source validation and archive anomalies

The current BCRA archive has two wrong-month targets:

- its May-2000 link serves a May-2001 document;
- its May-2011 link serves June 2011.

The original May-2000 BCRA document was recovered from a fixed Internet
Archive capture. The official archived index links `itemp0500.PDF`; the PDF
header says “as of May 31, 2000,” gross official reserve assets are USD
23,784 million and the selected predetermined drains are zero. The downloaded
8,137-byte payload is pinned to SHA-256
`a8b5b65682954f6c4318bb97707c3df2c633fd5c5108b5366cc381f234792c54`
([archived BCRA index](https://web.archive.org/web/20000819134455id_/http://www.bcra.gov.ar:80/english/contad/econ0100.htm),
[archived BCRA PDF](https://web.archive.org/web/20000916005139id_/http://www.bcra.gov.ar:80/pdfs/contad/itemp0500.PDF)).

For 2011-05, the current IMF IRFCL country-authority dataset supplies the same
template fields. The exact queried indicators are:

- `IRFCLDT1_IRFCL65_USD` — official reserve assets;
- `IRFCLDT2_IRFCL80_FO_USD` and `IRFCLDT2_IRFCL79_FO_USD` — principal and
  interest outflows;
- `IRFCLDT2_IRFCL1T_SHP_USD` — gross shorts;
- `IRFCLDT2_IRFCL48T_FO_USD`, `IRFCLDT2_IRFCL50T_FO_USD` and
  `IRFCLDT2_IRFCL46T_FO_USD` — repos, trade credit and payables.
- `IRFCLDT2_IRFCL78_USD` and `IRFCLDT2_IRFCL85_USD` — category-level
  reconciliation controls;
- `IRFCLDT2_IRFCL49T_IN_USD`, `IRFCLDT2_IRFCL50T_IN_USD` and
  `IRFCLDT2_IRFCL47T_IN_USD` — reverse-repo, trade-credit and
  other-receivable inflow diagnostics.

The query is monthly, country `ARG`, sector `S1X`, dataset
`IMF.STA:IRFCL`. The IMF history begins in 2000-M09 and is used only for the
missing 2011-05 publication. The raw IMF XML is archived and hashed alongside
the PDFs. For 2011-05, the row-3 detail sum is USD -664.78 million while the
official category aggregate `IRFCLDT2_IRFCL85_USD` is USD -665.55 million.
The USD -0.77 million difference is a reverse-repo inflow adjustment. Because
the selected construction subtracts gross outflow rows without netting
inflows, it uses the USD -664.78 million detail sum and retains the net
aggregate as a diagnostic. The resulting strict nominal NIR is approximately
USD 36,009.65 million.

The official IMF codelist verifies every mnemonic. For Argentina,
`IRFCLDT2_IRFCL46T_FO_USD`, `IRFCLDT2_IRFCL50T_IN_USD` and
`IRFCLDT2_IRFCL47T_IN_USD` return structurally valid datasets with zero
observations; they are not failed queries. At 2011-05, interest, shorts and
trade credit are also absent for that month. The replacement therefore records
four component-level zero assumptions (interest, shorts, trade credit and
payables) and marks the observation unsafe in the conservative variant.
Reverse-repo inflow diagnostic `IRFCLDT2_IRFCL49T_IN_USD` reports -0.77,
-5.24 and -2.85 million USD in 2011-05, 2011-06 and 2011-07, respectively;
these values explain the category/detail distinction without being subtracted
as gross outflows.

### Variants compared

1. **Strict NEDD — chosen.**
   `official reserve assets + negative principal/interest outflows
   + negative gross shorts + negative repo/trade/payable outflows`.
2. **BCRA-gross strict — retained.** Replace Section I.A with end-of-month
   BCRA ID `1` and keep the same outflows. This responds to the paper's
   wording “BCRA gross reserves,” but early levels do not always equal the
   template perimeter.
3. **Narrow daily proxy — retained only as robustness.**
   `ID1 - ID1243 - max(ID76,0)`. ID `76` is a passive foreign-currency repo
   with the exterior, not a China-swap series; the formula has no complete
   twelve-month maturity perimeter.
4. **Gross reserves alone — rejected.** It omits the defining liabilities.
5. **Broad analyst reserve measure — not selected.** Subtracting all deposits,
   the full swap or other headline liabilities can include amounts not due
   within twelve months and can double-count template items.

The constant-dollar conversion is:

`NIR_real_t = NIR_nominal_t * CPIAUCNS_2024-07 / CPIAUCNS_t`,

where July-2024 CPI-U is `314.540`. The series is therefore unchanged in the
base month.

### Hidden inconsistencies and final decision

- The template's “official reserve assets” perimeter and BCRA daily gross ID
  `1` are not identical in every month.
- Blank cells, zeroes and omitted observations need different treatment;
  parser methods are retained observation by observation.
- Server `Last-Modified` dates are not release dates.
- The template normally appears around day 20–23 of the following month, so
  NIR `t-1` can be unavailable at a day-15 nowcast origin.
- Archived documents can embody corrected rather than first-release values.

**Chosen construction:** strict monetary-authority NEDD, deflated to July-2024
US dollars. Under the explicitly flagged blank/absent-as-zero convention it is
continuous from 1999-12 through 2024-07 using 294 valid current-archive BCRA
files, one archived-original BCRA file (2000-05), and one official IMF
replacement (2011-05). The conservative variant is intentionally sparse.  
**Confidence:** medium.  
**Needed for exact replication:** BBVA's workbook, accounting perimeter and
item-level decisions, treatment of swaps/deposits/IMF positions, historical
pre-1999 data, and real-time vintages.

## Validation probes retained

| Probe or artifact | What it prevents |
|---|---|
| `research_probes/probe_cifra_ipc.py` | mistaking CIFRA for a fixed unweighted IPC-9 |
| `audits/historical_wage_transcription.csv` plus anchor assertions | decimal-comma/OCR scale errors and an unextended RIPTE splice |
| official 1991/1992 transcriptions and PCA loading audit | silently filling the 1989–1990 vehicle gap or using the wrong vehicle component |
| daily FX normalization and quote-count audit | treating outer metadata dates as continuous or mixing historical currency units |
| `research_probes/probe_nedd_parser.py` and `audits/nedd_parse_and_imf_reconciliation.csv` | selecting central-government columns, wrong archive months, valid-empty IMF detail responses or an incomplete liability row |
| raw and processed manifests | unrecorded source mutation; every payload/output has a SHA-256 hash |

## Final decision table

| Variable | Chosen construction | Rejected alternatives and reason | Unresolved assumptions | Achievable start date | Exact/public-proxy status | Confidence |
|---|---|---|---|---|---|---|
| CPI 2007–2015 | historical INDEC to 2006-12; three-province January-2007 bridge; CIFRA rates to 2016-12; national INDEC thereafter | official intervention CPI contradicts paper; fixed Mendoza panel has a four-year gap; Congreso lacks a stable source; endpoint rescale injects future information | author's components, weights, rate/level rule and vintage | 1965-01 in delivered panel | close public proxy, not exact | medium |
| ISBIC/RIPTE wages | linked non-qualified basic-agreement index through 1987; non-qualified ISBIC through 1994-07; ratio-linked RIPTE growth thereafter | annual/quarterly interpolation fabricates months; modern wage-index backcast changes concept; qualified/average categories lack author support | category, source edition, switch month, splice and vintage | 1965-02 for growth | close public proxy, not exact | medium |
| Pre-1993 activity | CPI-deflated BCRA codes 23+25, cars plus `resto`, crude steel; standardized log-level PC1 mapped to original EMAE; no 1989–1990 interpolation | code 22 mixes currencies/starts in 1990; code 23 alone is too narrow; dlog PCA misses reported level correlation; cars plus `utilitarios` is incomplete; interpolation is unsupported | exact credit/deflator, missing vehicles, PCA window, EMAE variant and mapping | 1965-04 for three-month growth; gap 1989-01–1991-03 | close public proxy; weak out-of-sample validation | low |
| Parallel FX | regime-specific official/financial/free/blue/CCL monthly means with audited denomination factors and explicit gaps | mechanical CCL or blue backcast is economically wrong; BCRA stale tail, Data912 live-only and unkeyed commercial FIEL series are unsupported | author's regime map, sources, quote side, monthly convention and boundaries | 1967-03 with gaps; continuous from 1990-06 | weak public proxy | low |
| Net international reserves | strict NEDD monetary-authority assets plus negative <=12m outflows; archived BCRA May-2000 and assumption-flagged IMF May-2011 recovery; CPI-U July-2024 dollars | gross alone omits liabilities; narrow IDs lack maturity completeness; broad measures risk double count and wrong horizon | blank/omitted-cell meaning, BBVA perimeter, swaps/deposits/IMF treatment and pre-1999 history | 1999-12, continuous only under flagged zero assumptions; conservative variant sparse | close public proxy, not BBVA-exact | medium |

## Implemented and validated snapshot

The authoritative retained implementation is processed snapshot
`20260725_final_v10`. Its retrieval manifest records 438 unique requests,
paths and SHA-256 hashes. The regenerable raw payload cache is intentionally
excluded from Git to avoid repository bloat. The output has 715 monthly rows
from 1965-01 through 2024-07 and 13 underlying variables. All 34 hard
validation checks pass. The additional warning is
deliberate: the selected activity proxy's 2014–2015 holdout correlation is
only 0.372 in log levels and 0.392 in monthly log growth, so the paper's
reported above-80% relationship is not treated as independently replicated.
