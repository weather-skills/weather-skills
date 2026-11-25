# Weather Skills

> ⚠️ **Under active development — not production ready.**
>
> These skills are an early experiment in tool composition for weather/climate
> data pipelines. Interfaces, dataset schema, and skill boundaries may change
> without notice. Fetchers hit real APIs and require credentials; middle-
> pipeline skills have only been smoke-tested on small synthetic data. Do not
> use in any automated workflow you rely on, and do not assume outputs are
> scientifically validated. Expect breakage.

Small tools an AI agent can run to transform weather and climate data. Each
skill is a command-line script that reads and writes one shared Zarr format, so
the output of one skill feeds the next — see [`STANDARD_DATASET.md`](https://github.com/weather-skills/weather-skills-core/blob/main/docs/weather-skill-authoring/references/STANDARD_DATASET.md). Data comes in
through fetcher skills, and figures come from the plot skill; the
[weather-skills catalog](https://github.com/weather-skills/weather-skills-catalog)
lists them.

Built as [Agent Skills](https://agentskills.io).

## Skills

### Generic middle (operate on any standard dataset)
| Skill | What it does |
|---|---|
| `clip-region` | Clip a gridded or station Zarr to a `--bbox N/W/S/E`, a named `--region`, or a `--geojson` polygon (use `resolve-region` for a country's bbox) |
| `aggregate-temporal` | Resample rates along `time`/`step` (mean/min/max) into fixed (`--period`) or rolling (`--window` duration) windows; duration-weights CF bounds; stamps `data_interval` (the period; with `--window`, the input spacing, and each window's `{dim}_bounds`), `aggregation_period`, `aggregation_coverage` and `cell_methods` |
| `convert-to-totals` | Terminal: rate × stamped `aggregation_period` → amount (100% coverage default; refuses overlapping Δt < period — `select` first) |
| `deaccumulate` | Convert a leftover cumulative-since-init forecast variable into per-step diffs along the `step` axis (fetchers already write rates) |
| `regularize-time` | Put overlapping or irregular `time`/`step` samples (e.g. one weekly mean per init) on a regular `--period` grid (`1 day`, `15 minute`): each cell is the mean of the samples covering it, with `aggregation_coverage` per cell; run before aggregate-temporal when samples overlap |
| `step-to-time` | Realize a forecast's `step` lead-time axis as wall-clock valid times (`time = init + step`) so it can be compared against time-based observations — one init, or many inits at one selected lead |
| `select` | Pick entries along one dim by position or coordinate value; a single pick collapses the dim (e.g. one forecast week from each model before `concat`) |
| `convert-calendar` | Convert the time axis to another CF calendar (`standard`, `noleap`, `360_day`, …) so two datasets share one calendar before comparing |
| `unit-convert` | Convert a variable to target `--to-units` (e.g. precip flux `kg m-2 s-1` → depth rate `mm/day` via ÷ liquid-water density) |
| `downscale` | Spatial downscaling onto a finer grid (by factor, finer resolution, or a reference grid) via `--algorithm` (`linear-interpolation` or `q-q` empirical quantile mapping) |
| `coarsen` | Coarsen or align a grid onto a target `(resolution, offset)` or another Zarr's grid, by linear interpolation or an area-weighted (conservative) mean — adds no information |
| `edit-coords` | Rename data variables, coordinates, or dims (`--rename OLD=NEW`) and re-index a dim by another 1-D coordinate (`--swap-dims OLD=NEW`); refuses edits that lose an ontology dim |
| `clip-values` | Limit values to `--min` / `--max` (e.g. `--min 0` to remove negative rainfall); NaN stays NaN |
| `drop` | Drop named data variables or non-index coordinates (`--name`) — e.g. stray `valid_time` / `spatial_ref` before `concat` |
| `concat` | Join Zarr stores along a named dim (incl. new dims with coord values) |
| `merge` | Combine the variables of several Zarrs into one dataset on their shared coordinates |
| `label-group` | Add label coords: the country / admin-1 / admin-2 / subregion / continent or GeoJSON polygon each cell or station lies in (`--level`, `--geojson`), and time labels (`--time month`, `year`, `dayofyear`, or month groups `MAM,OND`) — data unchanged |
| `summarize-dim` | Summarize named dims with a statistic (mean/std/min/max/sum/median/quantile) — e.g. ensemble spread as the std across `number`, a 10–90% band with `--method quantile -q 0.1 -q 0.9`, or a time-mean baseline. `--by LABEL` gives one value per label (per county, per season), e.g. after `label-group` |
| `point-value` | Sample a gridded dataset at point locations (station Zarr, `--point LAT,LON[,ID]`, or CSV) into a `station_id` point_obs dataset — nearest, bilinear, or N×N cell-mean — for grid-vs-station `difference` / `verify` |
| `local-extrema` | Find the local highs or lows of a gridded field (pressure centres, rainfall maxima) into a `station_id` point_obs dataset of locations and central values — for H / L map labels or listing centres |
| `csv-io` | Export a Zarr as it is to a CSV or Excel table (units in the headers), or read a CSV or Excel sheet — station records, a gridded table, or one gauge (`--point`) — into a Zarr. Day-first dates need `--date-format`; missing codes need `--na-value`. |
| `netcdf-io` | Export a Zarr to a NetCDF `.nc` file (attrs and history kept), or read one or more `.nc` files into a Zarr. Variables without units need `--units VAR=UNITS`. |
| `difference` | Subtract one dataset from another (A − B) with inner-join alignment and broadcasting — anomalies vs a baseline, scenario-minus-historical change maps |
| `climatology` | Daily climatology (mean, std, count per calendar day across years) computed from a multi-year daily Zarr, optional `--window` day pooling, laid onto a `--start-time`/`--end-time` window in `clim-fetch`'s layout — for a known base period or anything `clim-fetch` doesn't mirror |
| `standardize-anomaly` | Standardized anomaly aka z-score: `(field − clim_avg) / clim_std` against a climatology (e.g. `clim-fetch`) — dimensionless output, errors on units mismatch. For a plain physical-unit anomaly, use `difference` instead. |
| `zonal-moisture-transport` | Eastward moisture flux `q·u`, default column-integrated to IVT (`viwve`, kg m-1 s-1). Compose after `dynamical-fetch` (46-day `pressure_level` group) with `q` and `u` |
| `wind-speed-direction` | Wind speed (`hypot(u, v)`) and the directions the wind blows from and to (degrees clockwise from north) from a `--u` / `--v` component pair; calm cells (speed ≤ `--calm-speed`) get NaN direction |
| `verify` | Forecast verification for every lead: pairs forecast and obs (optional `--filter` masks), pools the samples into one score or keeps dims (`--keep`) or `--by` labels, optionally `--lat-weighted`, and writes RMSE, correlation, ACC, POD/FAR/ETS, Heidke, BSS, RPSS, spread, reliability and ROC, with the sample count behind each value. |
| `correlate` | Pearson or Spearman correlation of two datasets along a dim (default `time`), cell by cell, with p-value and pair count — forecast–obs anomaly correlation, or a climate index against a grid |
| `indicator` | Daily or weekly boolean indicator from one `--rule` (aliases `icpac-onset` / `chc-onset`, or clauses like `precip sum 8d >= 25mm`, thresholds in the data's units); optional `--probability`, `--detect first` / `any`, `--cumulative` |

### Agent capabilities
Capabilities the agent uses alongside pipelines; none of them produces a
dataset output.

| Skill | What it does |
|---|---|
| `resolve-region` | Resolve a country, multi-country region, ocean box, state/province/county, or landmark to a `--bbox N/W/S/E` (and optional boundary polygon GeoJSON, or with `--level admin_1` every county of a country in one file). |
| `resolve-time` | Resolve relative calendar dates ("the last two weeks", `latest`, `now-3d`) to absolute `--start-time`/`--end-time` or `--date`. Latest published day is the fetcher's `--probe-latest`, not this skill. |
| `inspect-zarr` | Print dims, coordinate values, and a bounded data-variable summary of a Zarr (stdout; no write). Data arrays can be huge — this skill never dumps them in full. |
| `provenance` | Inspect `weather_skills_history` on a Zarr, figure (PNG/JPEG/GIF/HTML/PDF/SVG), GeoJSON, or CSV/Excel table (DAG lineage, JSON, or a commit-pinned reproduction script). |

## Install

### As a CLI tool

For ad-hoc command-line use (no agent involved), install the skills as a
single `weather-skills` binary:

```bash
# One-shot, no install — list available skills
uvx --from git+https://github.com/weather-skills/weather-skills weather-skills

# Run one
uvx --from git+https://github.com/weather-skills/weather-skills weather-skills <skill> [args]
```

Or install once and invoke directly:

```bash
uv tool install git+https://github.com/weather-skills/weather-skills
weather-skills                          # list
weather-skills <skill> [args]           # run one
```

Each skill's PEP 723 inline dependency block is resolved by `uv run`
on each invocation, so the runner itself contributes no Python deps to the
script's runtime environment.

### As agent skills

For use by an LLM agent (Claude Code, etc.), install the `SKILL.md` files
into your project with [skillkit](https://github.com/rohitg00/skillkit) — no
local install needed, `npx` runs the latest skillkit on demand (add `@latest`
to always pull the newest):

```bash
# List what skillkit discovers in the repo
npx skillkit install weather-skills/weather-skills --list

# Install all skills to the current project
npx skillkit install weather-skills/weather-skills --all --yes

# Install globally so any project can use them
npx skillkit install weather-skills/weather-skills --all --yes --global

# Target a specific agent (otherwise skillkit installs for every agent it detects)
npx skillkit install weather-skills/weather-skills --all --yes --agent claude-code

# Install just a subset
npx skillkit install weather-skills/weather-skills --skill=clip-region
npx skillkit install weather-skills/weather-skills --skills=clip-region,difference

# Overwrite an existing install
npx skillkit install weather-skills/weather-skills --all --yes --force
```

Pin in a manifest for team / reproducible use:

```bash
npx skillkit manifest init
npx skillkit manifest add weather-skills/weather-skills
npx skillkit manifest install
```

See `npx skillkit install --help` for more flags.

## Composition pattern

Skills are designed to chain. Example — weekly rainfall totals from a daily
Zarr written by any fetcher:

```bash
weather-skills aggregate-temporal \
    --input /tmp/daily.zarr --period weekly --output /tmp/weekly.zarr
weather-skills convert-to-totals \
    --input /tmp/weekly.zarr --output /tmp/weekly_totals.zarr
```

In practice a user just states the goal in natural language and the agent
picks and composes skills from this set.

## Standard dataset contract

Skills share a simple Zarr contract: fixed dimension names (`latitude`,
`longitude`, `time`, `step`, `number`, `station_id`, …) and short types (`spatial`, `observations`, `forecast`,
`vertical_forecast`, `point_obs`, …).
See
[`STANDARD_DATASET.md`](https://github.com/weather-skills/weather-skills-core/blob/main/docs/weather-skill-authoring/references/STANDARD_DATASET.md).
Fetchers write that shape; other skills only depend on dims, coords, data
variables, and `weather_skills_*` attrs — not on per-variable encoding.

## CLI flag conventions

Each skill declares its CLI through the `@weather_skill` decorator from
`weather_skills_core`, so common parameters (`--input` / `-o`, `--bbox`,
`--start-time` / `--end-time`, etc.) mean the same thing wherever they appear.
See
[`CONVENTIONS.md`](https://github.com/weather-skills/weather-skills-core/blob/main/docs/weather-skill-authoring/references/CONVENTIONS.md)
for the full mapping of concept → canonical flag.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the publishing model (`main` is the
consumer-facing branch — every merge is a release), the PR workflow, and the
version-bump conventions (per-skill `metadata.version` in SKILL.md, driven by `release: major`
/ `release: minor` PR labels, with patch as the default).

## License

MIT. See [`LICENSE`](LICENSE).
