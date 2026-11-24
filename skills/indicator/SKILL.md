---
name: indicator
description: "Apply a yes/no weather rule, such as more than 25 mm of rain in 8 days or a rainy-season onset definition (ICPAC, CHC), to each day or week of a daily or weekly Zarr, and optionally reduce it to an onset date or an ensemble probability. Use for onset dates, wet or dry spells, and the chance that a threshold is met."
license: MIT
compatibility: Requires Python 3.12 and uv.
allowed-tools: Bash(uv run ${CLAUDE_SKILL_DIR}/scripts/indicator.py *)
metadata:
  version: "0.0.11"
  catalog-group: transforms
---

# indicator

Writes a 0/1 mask from a `--rule`, evaluated on each date of a series (or once on a static field), then optional reductions. Thresholds are in the data's own units; nothing is converted.

## Usage

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/indicator.py -i IN.zarr -o OUT.zarr --rule RULE [-v NAME] [--time-dim DIM] [--detect first|any] [--cumulative] [--first-within WINDOW] [--probability]
```

`RULE` is an alias or clauses joined by all `and` or all `or`:
```text
[not] VAR OP THR [after Nd | within Nd]
[not] VAR AGG [DAILY_THR] WINDOW [OP THR] [after Nd | within Nd]
```
- `THR` and `DAILY_THR` are in the variable's own units, written right after the number with no spaces: `20mm`, `5mm/day`, `35degC`, or a bare number on a unitless field (`land_sea_mask >= 0.5`). Any spelling of the same units matches (`mm/day` = `mm day-1`). The `THR` after `count-above`/`count-below` is a number of steps and takes no units.
- `VAR OP THR` (e.g. `tp > 25mm`): each step's own value, so on weekly totals it is the week's total. Same as `VAR sum 1d` on daily data or `VAR sum 1w` on weekly data.
- `AGG`: `sum`, `mean`, `count-above`, `count-below` (number of steps above/below `DAILY_THR`), `consecutive-above`, `consecutive-below` (every step in the window above/below `DAILY_THR`; no `OP THR`).
- `WINDOW`: `Nd` or `Nw`, starting on the labelled date and looking forward. `OP`: `>=`, `<=`, `>`, `<`.
- `after Nd`: evaluate the clause N days later. `within Nd`: true if the clause is true on any of the next N days (days 1 to N after the date). `not` inverts the clause.
- There is no `then`: a later condition is a clause with `after` or `within`. A false-start check is an onset clause `and not` a dry spell `within` the days that follow, e.g. onset of at least 20 mm in 3 days on at least 2 wet days, with no 7-day dry spell starting in the next 10 days: `precip sum 3d >= 20mm and precip count-above 1mm 3d >= 2 and not precip consecutive-below 1mm 7d within 10d`. To keep the whole spell inside the next 10 days, use `within 4d` (spells starting on days 1–4).
- Aliases: `icpac-onset` = `precip sum 3d >= 20mm and not precip consecutive-below 1mm 7d within 21d`; `chc-onset` = `precip sum 10d > 25mm and precip sum 20d > 20mm after 10d`. Both need daily precipitation totals in mm.

### Arguments
- `--input`, `-i` — daily or weekly Zarr on `time` or `step`. A single selected step or date (a scalar `step` or `time`, e.g. week 2 only) works; a window on it needs `data_interval` stamped. Only windows, `after`/`within` and `--first-within` need daily or weekly steps: plain comparisons (`VAR OP THR`) run at any step (e.g. hourly, also with `--detect` or `--cumulative`), and with no time axis at all on a static field such as a land-sea fraction.
- `--output`, `-o` — output Zarr.
- `--rule` — alias or clause string. Required.
- `--variable`, `-v` — use this variable in every clause instead of the one written in the rule.
- `--time-dim` — axis to evaluate along. Default: `time` if it has more than one value, else `step`.
- `--detect` — collapse the axis: `first` (date of the first true day) or `any` (true at least once).
- `--cumulative` — once true, stay true.
- `--first-within WINDOW` — window such as `7d` or `1w`. Marks the first true day of the input and the `WINDOW − 1` dates before it as 1, earlier dates as 0, and later dates as NaN: each date answers "does the first event fall within WINDOW from here?". Combines with none of the other reductions.
- `--probability` — fraction of ensemble members (`number`) where true; unchanged if there is no `number`.

### Output
- Default: `indicator` (0/1/NaN, `units: 1`) on the input dims.
- `--detect first`: `indicator_time` (first true date or step) and, on a datetime axis, `indicator_doy` (day of year).
- `--detect any`: `indicator` without the time axis.
- `--probability`: `probability` (0–1, `units: 1`).
- `--first-within`: `indicator` is a buffer of 1s ending on the first true day. With `--first-within 3d` and the first true day on day 6: days 1–3 are 0, days 4–6 are 1, day 7 onward is NaN. A gap before the first true day makes dates whose window reaches it NaN, and every date after it, since the event could be hidden in it; a window holding both the gap and the first true day is 1. A window running past the end of the data is 1 if the first true day is in it, else NaN. `aggregation_period` is the window.
- Each variable has `long_name` and `indicator_spec` (parsed rule as JSON); the store has `indicator_rule` and `indicator_expanded`.

### Notes
- Values are compared as stored; nothing is converted. A threshold in other units, or a bare number on a variable with units, exits 2. Convert the data first: `unit-convert` for other units of the same quantity (e.g. K to degC), `convert-to-totals` for rates to amounts. Daily rates from a fetcher need `aggregate-temporal --period daily` before `convert-to-totals`.
- Evaluate on individual members, not an ensemble mean. Windows must be whole steps (`1w`, not `10d`, on weekly data).
- A variable without the time axis (a static field next to a daily series) takes only plain comparisons; it holds for every date.
- A missing value makes every window that covers it NaN. `--detect first` gives no date when the rule is never true or when a gap comes before the first true day. `--detect any` is NaN when a gap could hide the event. `--probability` is NaN where any member is NaN.
- The last dates of the input cannot be decided (27 days for `icpac-onset`). stderr prints the last decidable date; NaN after it means "not yet known".
- `--detect` cannot be combined with `--cumulative`, and `--detect first` cannot be combined with `--probability`.
- `--first-within` counts from the first date of the input, so start the input at the season start: an event before it is not seen.

### Example
```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/indicator.py -i ecmwf_s2s_kenya_daily_mm.zarr -o onset_p.zarr --rule icpac-onset --cumulative --probability
```
Writes, for each lead day of the 2026-10-12 forecast over Kenya (daily totals in mm), the fraction of members whose ICPAC onset has happened by then.

## Details

| Want | Flags |
| --- | --- |
| Daily P(window exceeded) | `--probability` |
| Onset date per member | `--detect first` |
| P(onset has happened by each date) | `--cumulative --probability` |
| P(event happens at least once) | `--detect any --probability` |
| Observed onset within the week from each date, post-onset dates left out | `--first-within 7d` |

`indicator_doy` wraps at 1 January, so a mean across members whose onsets straddle the year end is wrong. `and` / `or` stay decided when one side alone decides them (a false side makes `and` false even if the other is NaN). For calendar onset dates from a forecast, convert `step` to valid `time` first.

Land mask from a static land-sea fraction (e.g. ERA5 `land_sea_mask`, fetched for one day): keep one time with `select --dim time --index 0`, remove that time with `drop --name time`, then `indicator --rule 'land_sea_mask >= 0.5'`. The output has no time axis, so `verify --filter land.zarr` applies it to every date and lead. Put it on the scoring grid first (`coarsen` or `downscale`) if the grids differ.
