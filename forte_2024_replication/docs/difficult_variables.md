# Critical audit of the five difficult variables

This file records the corrected working conclusions. The independent evidence,
alternatives, rejection reasons, unresolved assumptions, and final decision
table are in
[`difficult_variables_final.md`](difficult_variables_final.md).

## 1. Argentine CPI, 2007–2015

The paper does not disclose the component series, weights, missing-data rule,
or whether its average was formed from levels or rates. No stable primary
machine-readable Congreso series was found.

Institutional chronologies place the old IPC Congreso sequence from May 2011
through October 2015; successor-group releases for November 2015–January 2016
are not demonstrably the same series. CABA's linked level begins in July 2012,
but the first located IPCBA release was September 2013 for August 2013, making
its earliest linked observations retrospective. The downloaded CIFRA
workbook is likewise a current reconstruction with no historical workbook
vintages.

The public baseline is:

1. official linked INDEC CPI through December 2006;
2. an explicitly assumed January-2007 bridge equal to the arithmetic mean of
   the simple Mendoza, Neuquén, and San Luis rates;
3. monthly rates from CIFRA's institutional **IPC Provincias** workbook from
   February 2007 through December 2016;
4. official national CPI rates from January 2017.

CIFRA is not a fixed “IPC-9.” Its methodology changes provincial composition
and imputes discontinued components. Mendoza also has a 48-month internal gap
from April 2012 through March 2016, so it is not used in the continuous-panel
sensitivity. The preferred sensitivity is the equal mean of Neuquén and San
Luis rates.

The chain is not endpoint-rescaled: forcing a December-2016 match would alter
inflation or introduce information unavailable during the intervention.

Primary evidence:
[CIFRA workbook and methodology](https://centrocifra.org.ar/estadisticas/ipc-provincias/),
[INDEC archive warning](https://www.indec.gob.ar/indec/web/Institucional-Indec-InformacionDeArchivo-1),
and the [official API](https://apis.datos.gob.ar/series/api/series).

## 2. ISBIC/RIPTE wage splice

The original audit incorrectly concluded that monthly pre-1988 wages were
unavailable. INDEC's digitized 1991 volume contains monthly general
basic-agreement indices for qualified and non-qualified workers from 1945
through 1989; the build extracts 1965–1987 and switches to the later official
table in 1988. The volume explicitly links the historical and base-1988
scales. The build uses
**Personal No Calificado**, the common pension-index convention, but Forte
does not disclose the category.

The series is:

- archival linked non-qualified index through 1987-12;
- official monthly ISBIC non-qualified index from 1988-01 through 1994-07;
- RIPTE ratio-linked at 1994-07, with RIPTE growth from 1994-08.

ISBIC measures collective-agreement basic pay; RIPTE measures taxable
remuneration of stable registered workers. A ratio link removes the level
jump, not the conceptual break. The early series is a deterministic machine
extraction with raw tokens and correction methods retained: 265 of 276
baseline cells use automatic OCR normalization and 11 use manually inspected
overrides. It has not been independently double-entered. OCR-derived
annual-mean diagnostics differ by 3.79 percent in 1977 and 1.89 percent in
1986, apparently because the mean-row OCR is itself defective, so those mean
checks are not conclusive validation.

RIPTE is not a real-time series back to 1994. Resolution 2/2018 says it had
been calculated and published since 2006, making 1994–2005 retrospective.
Its official lag is up to 45 days, so final `t-1` is generally unavailable at
a day-15 origin.

Primary evidence:
[INDEC 1991 wage volume](https://biblioteca.indec.gob.ar/bases/minde/4si9_4.pdf),
[official ISBIC table](https://www.argentina.gob.ar/sites/default/files/indice_isbic.pdf),
[ISBIC methodology](https://www.argentina.gob.ar/sites/default/files/informe_isbic.pdf),
and RIPTE ID `158.1_REPTE_0_0_5`.

## 3. Pre-1993 activity PCA

Outer metadata coverage was misleading:

- both automobile API components are null from 1989-01 through 1992-12;
- crude steel is null from 1991-01 through 1992-12.

Official archival publications recover monthly total vehicles and crude steel
for 1991 and revised 1992. They do not provide defensible complete monthly
vehicle totals for 1989–1990. Those 24 months remain missing.
Before inserting those observations, the build verifies both that the API
patch cells remain null and that the API equals the separately archived
official CSV in values and missingness. The 1991 monthly steel rows sum to
2,972.0 thousand tonnes versus the printed 2,972.3 total; the 0.3 difference
is disclosed and not allocated to a month. The selected cars-plus-`resto`
series ends in 2015-12, which is sufficient for the 2014–2015 holdout but not
for a post-2015 physical-input continuation.

The preferred total-vehicle definition is automobiles plus API `resto`, not
automobiles plus the narrower `utilitarios` field. The credit catalogue shows
that code `23` is private-nonfinancial-sector peso credit and code `25` is
public-nonfinancial-sector peso credit. Their matched-unit sum is the chosen
public reading of Forte's “crédito total en pesos.” Code `23` alone is retained
as a sensitivity. Code `22` is rejected because it mixes peso and
foreign-currency credit and begins only in 1990.

The baseline PCA:

1. sums codes `23+25` and deflates the result with the linked CPI;
2. logs real credit, total vehicles, and crude steel;
3. standardizes and estimates PC1 over 1993-01–2013-12;
4. fixes its sign against original NSA EMAE;
5. maps PC1 to log EMAE in the same overlap and anchors January 1993;
6. uses the proxy through 1992 and linked EMAE thereafter;
7. computes simple monthly growth and a trailing three-month mean.

The resulting in-sample log-level correlation with EMAE is about 0.874, but
monthly-growth correlation is about 0.526. A natural 2014–2015 post-fit
holdout falls to about 0.372 in log levels and 0.392 in monthly log growth.
The narrower code-23-only sensitivity reaches about 0.924 in-sample levels.
The paper's above-80-percent statement can therefore be matched only in
trend-dominated in-sample levels. The primary PCA is retrospective and not
vintage-safe. A 1965–1988 loading sensitivity is retained, but its EMAE
sign/mapping still uses future data.

Primary evidence:
[official automotive CSV](https://infra.datos.gob.ar/catalog/sspm/dataset/330/distribution/330.3/download/datos-historicos-industria-automotriz-unidades-mensuales.csv),
[official steel CSV](https://infra.datos.gob.ar/catalog/sspm/dataset/359/distribution/359.3/download/datos-historicos-de-la-industria-siderurgica-datos-mensulaes.csv),
[1991 Ministry tables](https://cdi.mecon.gov.ar/greenstone/collect/iet/index/assoc/HASH0b19.dir/doc.pdf),
and [revised 1992 INDEC tables](https://biblioteca.indec.gob.ar/bases/minde/epidic93.pdf).

## 4. Historical parallel exchange rate

The BCRA historical series contain non-null blocks and undocumented currency
rescalings inside their advertised date spans. They are normalized before
monthly aggregation. Controlled-period gaps remain missing, and boundary
months carry quote-count/first-date/last-date/partial-month diagnostics.

The baseline regime rule is:

| Period | Selected market |
|---|---|
| 1965-01–1967-02 | unavailable |
| 1967-03–1971-08 | official rate, documented unified market |
| 1971-09–1975-12 | BCRA financial sale |
| 1976-01–1980-12 | BCRA free sale |
| 1981-01–1981-05 | official, unified interval |
| 1981-06–1981-12 | BCRA financial sale |
| 1982-01–1982-06 | unavailable |
| 1982-07–1982-12 | BCRA financial sale |
| 1983-01–1987-01 | BCRA free sale |
| 1987-02–1987-10 | unavailable |
| 1987-11–1989-04 | BCRA free sale |
| 1989-05–1990-01 | unavailable; stale/mixed-denomination observations rejected |
| 1990-02–1990-05 | low-confidence normalized BCRA free sale |
| 1990-06–2011-09 | official, unified market |
| 2011-10–2012-12 | Ámbito blue sale |
| 2013-01–2015-12 | Ámbito/Rava CCL |
| 2016-01–2019-08 | official, unified market |
| 2019-09–2024-07 | Ámbito/Rava CCL |

Ámbito's endpoint backfills blue to January 2002, but existence does not make
blue the economically relevant rate in a unified regime. `data912` supplies
live MEP/CCL only and was rejected for history. FIEL's underlying monthly
DataFIEL series is institutional but commercial; the unrelated FIEL book URL
in the first audit was removed.

The machine-readable audit separates ten carrier-source transitions from
four additional observed-block starts after gaps. It also retains a
same-source growth sensitivity that is missing only at carrier changes, plus
an explicit flag for unified-market gaps imposed to zero.

Primary evidence:
[official historical FX dataset](https://datos.gob.ar/dataset/sspm-tipos-cambio-historicos/archivo/sspm_175.1),
[FIEL's exchange-control study](https://www.fiel.org/publicaciones/IndicadoresCoyuntura/COYU_99_1662770144699.pdf),
and [Ámbito's historical page](https://www.ambito.com/contenidos/dolar-informal-historico.html).

## 5. Net international reserves

The BBVA Research estimate is unavailable. BCRA ID `76` is a passive USD repo
series, not the China swap; therefore `ID1-ID1243-ID76` is rejected as the
baseline.

The chosen public construction uses the monetary-authority columns in the
BCRA/IMF reserves-liquidity template:

`official reserve assets + signed-negative principal/interest outflows <=12m
 + gross short positions as liabilities
 + signed-negative repo, trade-credit and payable outflows`.

The parser does not blindly negate every positive cell. It has two
independently reconciled omitted-minus overrides: February-2005 capital and
November-2008 repos.

The parser:

- separates monetary-authority from central-government columns;
- handles Spanish and English templates, historical header typos, and changing
  number formats;
- checks the embedded reference month against the archive month;
- distinguishes reported values/explicit zeroes from blank- or
  absent-as-zero assumptions for every selected liability category;
- retains NEDD-gross and BCRA-ID1-gross variants;
- deflates with `CPIAUCNS`, July 2024 = 314.540.

The continuous baseline uses the official-template convention that blank or
absent cells are nil, with an observation-level flag. This affects 264 of 296
months, driven mainly by the short-position row (257 blank and seven absent).
Row 1 is absent-assumed-zero in 16 months and row 3 in 80. A conservative
variant sets NIR to `NA` whenever any such assumption is needed.

The current BCRA archive serves wrong-month files for 2000-05 and 2011-05.
The original May-2000 BCRA publication was independently recovered from the
Internet Archive capture `20000916005139`; its frozen 8,137-byte payload has
SHA-256 `a8b5b65682954f6c4318bb97707c3df2c633fd5c5108b5366cc381f234792c54`.
Official IMF SDMX recovers 2011-05 from the same country-authority template.
The official codelist verifies every queried mnemonic. For Argentina,
payable-outflow `IRFCLDT2_IRFCL46T_FO_USD` and the two inflow diagnostics
`IRFCLDT2_IRFCL50T_IN_USD` and `IRFCLDT2_IRFCL47T_IN_USD` are valid but
contain zero observations. May-2011 interest, shorts, trade credit and
payables are therefore explicit zero assumptions, not reported zeroes.
Reverse-repo inflow `IRFCLDT2_IRFCL49T_IN_USD` reports -0.77, -5.24 and
-2.85 million USD in May, June and July 2011 and is retained as a
reconciliation diagnostic rather than subtracted as an outflow.
No baseline month remains missing from 1999-12 through 2024-07 under that
explicit reporting convention. July 2024 yields
approximately –USD 6.493 billion before deflation (July is the base month).

Primary evidence:
[BCRA NEDD archive](https://www.bcra.gob.ar/normas-especiales-para-la-divulgacion-de-datos-fmi/),
[IMF IRFCL dataset](https://data.imf.org/en/datasets/IMF.STA:IRFCL),
[archived May-2000 BCRA document](https://web.archive.org/web/20000916005139id_/http://www.bcra.gov.ar:80/pdfs/contad/itemp0500.PDF),
and [BCRA reserves data](https://www.bcra.gob.ar/catalogo_de_datos/reservas-internacionales-del-bcra-y-base-monetaria/).
