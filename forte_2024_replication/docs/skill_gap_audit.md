# Skill-gap audit

## Installed skills inspected and used

The complete `SKILL.md` for each installed project skill and its relevant
references/scripts/assets were inspected before source selection.

| Skill | Material use in this project | Limitation exposed by the audit |
|---|---|---|
| `indec` | verified IDs/metadata and fetched CPI, provincial CPI, RIPTE, automobile, steel, EMAE and historical exchange-rate data from the official Argentina time-series API | latest-data API has no complete release-vintage archive and contains material null blocks |
| `bcra-macro` | verified the BCRA v4 catalogue and IDs; guided paginated API retrieval and the historical `panser.txt` catalogue/file | v4 alone does not provide the 1960s monetary history or BBVA's private NIR perimeter |
| `fred-macro` | authenticated FRED metadata and observations using the environment `FRED_API_KEY`; supplied CPI-U, official FX and commodity controls | FRED is an institutional republisher for some series and cannot recover Argentine private histories |
| `data912` | tested current MEP/CCL endpoints and response schema | current quotes only; no usable long historical route was exposed |

No key is stored in code, a manifest or output. The raw manifest redacts the
FRED API key.

## `gauss314/skills` catalogue review

The catalogue and the only two plausible Argentine-market candidates were
inspected:

- [`mae`](https://github.com/gauss314/skills/blob/main/skills/mae/SKILL.md)
  uses public but undocumented MAE endpoints. It requires `requests`, no API
  key, and can support modern wholesale-market validation. It does not supply
  a homogeneous historical blue/free/CCL series from 1965.
- [`byma`](https://github.com/gauss314/skills/blob/main/skills/byma/SKILL.md)
  uses an undocumented BYMA endpoint, `requests`, no API key, and instrument
  OHLCV. It could construct a modern bond-implied CCL only while suitable ARS,
  USD and C species coexist. Instrument turnover, changing liquidity,
  endpoint instability and disabled TLS validation in the catalogue script
  are material reliability risks.

Global market skills such as Alpha Vantage, Yahoo Finance or Investing were
also rejected as redundant. FRED/EIA/IMF and the World Bank Pink Sheet are
more authoritative for oil and wheat and have the required long history.

## Recommendation

**No additional skill is recommended or installed.**

Neither `mae` nor `byma` closes a remaining replication gap:

- the CPI gap is the author's undisclosed averaging/vintage rule, not access
  to another market API;
- the wage series is now available from official publications, while its
  remaining ambiguity is the author's category and splice;
- activity is blocked by 1989–1990 monthly vehicle observations and
  undisclosed PCA choices;
- parallel FX is blocked by the author's historical source/regime map, not a
  lack of another modern CCL provider; and
- exact NIR requires BBVA's accounting workbook and historical decisions.

Installing a new skill would therefore add a redundant or unstable endpoint
without materially improving the defensible start date or exact-replication
status. If a later robustness exercise specifically requests transaction-
based CCL after 2020, `byma` could be reconsidered as a validation source—not
as a solution to this replication's historical gap.
