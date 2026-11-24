# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "weather-skills-core>=0.4.0",
# ]
# ///
"""Apply a yes/no weather rule to each day or week of a Zarr, optionally reduced to an onset date or a probability."""

import json
import re
import sys
from dataclasses import asdict, dataclass

import numpy as np
import xarray as xr
from weather_skills_core import Dataset, UsageError, weather_skill
from weather_skills_core.units import (
    data_interval_of,
    infer_timestep,
    parse_aggregation_period,
    units_equal,
    ureg,
    variable_units,
)

# Auto-populated by the version-bump CI workflow. Do not edit manually.
_SKILL_VERSION = "0.0.11"


# --- --rule parsing ---------------------------------------------------------

ALIASES = {
    "icpac-onset": "precip sum 3d >= 20mm and not precip consecutive-below 1mm 7d within 21d",
    "chc-onset": "precip sum 10d > 25mm and precip sum 20d > 20mm after 10d",
}

_AGGS = (
    "sum",
    "mean",
    "count-above",
    "count-below",
    "consecutive-above",
    "consecutive-below",
)
_COUNT_OR_CONSEC = frozenset(
    {"count-above", "count-below", "consecutive-above", "consecutive-below"}
)
_CONSEC = frozenset({"consecutive-above", "consecutive-below"})
_EXAMPLES = (
    "Examples: 't2m > 35degC' (each step's own value), 'precip sum 8d >= 25mm', "
    "'precip count-above 1mm 10d >= 5', 'precip consecutive-below 1mm 7d' (7 dry days in a "
    "row), 'precip sum 3d >= 20mm and not precip consecutive-below 1mm 7d within 21d' (onset "
    "with no false start), 'land_sea_mask >= 0.5' (no units on a unitless field)."
)
_WINDOW_RE = re.compile(r"^(\d+)([dw])$", re.IGNORECASE)
_SPLIT_RE = re.compile(r"\s+(and|or)\s+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(\S*)$")


@dataclass(frozen=True)
class Clause:
    negate: bool
    variable: str
    agg: str
    window: int
    op: str | None
    threshold: float | None
    daily_threshold: float | None
    units: str | None  # units of the threshold compared against the data
    after: int | None
    within: int | None


@dataclass(frozen=True)
class IndicatorSpec:
    source: str
    expanded: str
    combinator: str | None
    clauses: tuple[Clause, ...]

    def to_json(self) -> dict:
        return asdict(self)


def parse_rule(raw: str) -> IndicatorSpec:
    """Parse a named alias or a clause string joined by ``and`` / ``or``."""
    if not isinstance(raw, str) or not raw.strip():
        raise UsageError("--rule is required (a named alias or a clause string)")
    source = raw.strip()
    key = source.lower()
    expanded = ALIASES.get(key, source)
    parts = _SPLIT_RE.split(expanded.strip())
    if not parts or not parts[0].strip():
        raise UsageError(f"could not parse --rule {source!r}")
    clauses = [parts[0]]
    combinators = []
    for i in range(1, len(parts), 2):
        combinators.append(parts[i].lower())
        if i + 1 >= len(parts) or not parts[i + 1].strip():
            raise UsageError(f"could not parse --rule {source!r}")
        clauses.append(parts[i + 1])
    unique = set(combinators)
    if len(unique) > 1:
        raise UsageError(
            "mixing 'and' and 'or' in one --rule is not supported; use a single combinator"
        )
    parsed = tuple(_parse_clause(c.strip(), source) for c in clauses)
    combinator = combinators[0] if combinators else None
    return IndicatorSpec(source=source, expanded=expanded, combinator=combinator, clauses=parsed)


def _parse_window(token: str, source: str) -> int:
    match = _WINDOW_RE.fullmatch(token)
    if not match:
        raise UsageError(f"window {token!r} in --rule {source!r} must look like '8d' or '1w'")
    days = int(match.group(1)) * (7 if match.group(2).lower() == "w" else 1)
    if days < 1:
        raise UsageError(f"window must be >= 1d in --rule {source!r}")
    return days


def _parse_op(token: str, source: str) -> str:
    if token not in _OPS:
        raise UsageError(
            f"comparator {token!r} in --rule {source!r} must be one of {', '.join(_OPS)}"
        )
    return token


def _parse_quantity(token: str, source: str, what: str) -> tuple[float, str | None]:
    """Split '20mm' or '0.5' into a number and its units (None when none are written)."""
    match = _NUMBER_RE.fullmatch(token)
    if not match:
        raise UsageError(
            f"{what} {token!r} in --rule {source!r} must be a number with optional units "
            "written without spaces, e.g. '20mm', '5mm/day', '35degC' or '0.5'"
        )
    units = match.group(2) or None
    if units is not None:
        try:
            ureg.Unit(units)
        except Exception as exc:  # noqa: BLE001
            raise UsageError(
                f"unknown units {units!r} in {what} {token!r} of --rule {source!r}"
            ) from exc
    return float(match.group(1)), units


def _parse_clause(text: str, source: str) -> Clause:
    tokens = text.split()
    if not tokens:
        raise UsageError(
            f"empty clause in --rule {source!r}. Known aliases: {', '.join(sorted(ALIASES))}"
        )
    i = 0
    negate = False
    if tokens[i] == "not":
        negate = True
        i += 1
    if i >= len(tokens):
        raise UsageError(f"clause {text!r} in --rule {source!r} is missing a variable")

    variable = tokens[i]
    i += 1
    if i < len(tokens) and tokens[i] in _OPS:
        # Plain threshold, e.g. 'tp > 25': each step's own value (window 0 = one step).
        op = _parse_op(tokens[i], source)
        if i + 1 >= len(tokens):
            raise UsageError(f"clause {text!r} in --rule {source!r} is missing a threshold")
        threshold, units = _parse_quantity(tokens[i + 1], source, "threshold")
        after, within = _parse_tail(tokens, i + 2, text, source, "value")
        return Clause(
            negate=negate,
            variable=variable,
            agg="value",
            window=0,
            op=op,
            threshold=threshold,
            daily_threshold=None,
            units=units,
            after=after,
            within=within,
        )
    if i >= len(tokens) or tokens[i] not in _AGGS:
        aliases = ", ".join(sorted(ALIASES))
        raise UsageError(
            f"could not parse clause {text!r} in --rule {source!r}. "
            f"Expected '{{variable}} {{op}} {{threshold}}' or '{{variable}} {{agg}} …' with agg "
            f"one of {', '.join(_AGGS)}. Known aliases: {aliases}. {_EXAMPLES}"
        )
    agg = tokens[i]
    i += 1

    daily_threshold = None
    units = None
    if agg in _COUNT_OR_CONSEC:
        if i >= len(tokens):
            raise UsageError(
                f"clause {text!r} in --rule {source!r} needs a daily threshold after {agg}"
            )
        daily_threshold, units = _parse_quantity(tokens[i], source, "daily threshold")
        i += 1

    if i >= len(tokens):
        raise UsageError(f"clause {text!r} in --rule {source!r} is missing a window (e.g. 8d)")
    window = _parse_window(tokens[i], source)
    i += 1

    op = None
    threshold = None
    if agg not in _CONSEC:
        if i >= len(tokens):
            raise UsageError(
                f"clause {text!r} in --rule {source!r} needs a comparator and threshold"
            )
        op = _parse_op(tokens[i], source)
        i += 1
        if i >= len(tokens):
            raise UsageError(f"clause {text!r} in --rule {source!r} is missing a threshold")
        threshold, threshold_units = _parse_quantity(tokens[i], source, "threshold")
        if agg in _COUNT_OR_CONSEC and threshold_units is not None:
            raise UsageError(
                f"threshold {tokens[i]!r} in --rule {source!r} counts steps, so it takes no "
                f"units; put units on the daily threshold after {agg}"
            )
        if agg not in _COUNT_OR_CONSEC:
            units = threshold_units
        i += 1

    after, within = _parse_tail(tokens, i, text, source, agg)
    return Clause(
        negate=negate,
        variable=variable,
        agg=agg,
        window=window,
        op=op,
        threshold=threshold,
        daily_threshold=daily_threshold,
        units=units,
        after=after,
        within=within,
    )


def _parse_tail(tokens: list[str], i: int, text: str, source: str, agg: str):
    """Parse the optional ``after Nd`` / ``within Nd`` qualifiers from ``tokens[i:]``."""
    after = None
    within = None
    while i < len(tokens):
        tag = tokens[i]
        if tag not in ("after", "within"):
            hint = ""
            if tag in _OPS and agg in _CONSEC:
                hint = (
                    " consecutive-above/-below take no comparator: they are true when every "
                    "step in the window is above/below the daily threshold. For 'at least N "
                    "steps' use count-above/-below with '>= N'."
                )
            elif tag == "then":
                hint = (
                    " Clauses join with 'and' or 'or'; for something that must follow, put "
                    "'after Nd' or 'within Nd' on the later clause."
                )
            raise UsageError(
                f"unexpected {tag!r} in clause {text!r} of --rule {source!r}.{hint} {_EXAMPLES}"
            )
        if i + 1 >= len(tokens):
            raise UsageError(f"{tag} in --rule {source!r} needs a window (e.g. {tag} 10d)")
        days = _parse_window(tokens[i + 1], source)
        if tag == "after":
            if after is not None:
                raise UsageError(f"repeated 'after' in clause {text!r} of --rule {source!r}")
            after = days
        else:
            if within is not None:
                raise UsageError(f"repeated 'within' in clause {text!r} of --rule {source!r}")
            within = days
        i += 2

    if after is not None and within is not None:
        raise UsageError(
            f"clause {text!r} in --rule {source!r} cannot take both 'after' and 'within'"
        )
    return after, within


# --- evaluation and reductions ------------------------------------------------

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}


def evaluate_spec(
    ds, spec, dim: str, variable_override: str | None, step_days: int = 1
) -> xr.DataArray:
    """Return a 0/1/NaN mask aligned to ``dim`` (ensemble ``number`` kept).

    Clause windows are in days; ``step_days`` is the spacing of ``dim``.
    """
    masks = []
    order = []  # input dims, in an order every variable the rule reads agrees with
    for clause in spec.clauses:
        name = variable_override or clause.variable
        if name not in ds:
            raise UsageError(
                f"variable {name!r} missing from --input. Available: {list(ds.data_vars)}"
            )
        _check_units(ds[name], clause.units, source=spec.source)
        dims = ds[name].dims
        for i, d in enumerate(dims):
            if d not in order:
                later = [order.index(x) for x in dims[i + 1 :] if x in order]
                order.insert(min(later, default=len(order)), d)
        masks.append(_evaluate_clause(ds[name], clause, dim, step_days))
    out = masks[0]
    combine = spec.combinator or "and"
    for other in masks[1:]:
        out = _combine(out, other, combine)
    out = out.transpose(*(d for d in order if d in out.dims), ...)
    return out.astype("float32")


def apply_reductions(
    mask: xr.DataArray,
    dim: str,
    *,
    cumulative: bool,
    detect: str | None,
    probability: bool,
    first_within: int | None = None,
) -> xr.Dataset:
    """Apply ``--first-within`` or ``--cumulative``, ``--detect``, ``--probability`` in that order.

    ``first_within`` is the window length in steps of ``dim``.
    """
    if first_within is not None and (cumulative or detect or probability):
        raise UsageError(
            "--first-within cannot be combined with --cumulative, --detect or --probability; "
            "for a forecast's chance of the event by each lead use --cumulative --probability"
        )
    if detect is not None and cumulative:
        raise UsageError("--detect cannot be combined with --cumulative")
    if detect == "first" and probability:
        raise UsageError(
            "--detect first writes dates, not 0/1; use --detect any --probability "
            "or --cumulative --probability"
        )

    field = mask
    if first_within is not None:
        field = _first_within(field, dim, first_within)
    if cumulative:
        field = _cumulative(field, dim)

    if detect == "first":
        return _detect_first(field, dim)
    if detect == "any":
        field = _detect_any(field, dim)

    if probability:
        field = _probability(field)
        name = "probability"
    else:
        name = "indicator"
    field = field.astype("float32")
    field.name = name
    return field.to_dataset()


def _check_units(da: xr.DataArray, units: str | None, source: str) -> None:
    """Refuse a threshold whose units differ from the data's; nothing is converted."""
    data_units = variable_units(da)
    if data_units is not None:
        data_units = format(ureg.Unit(data_units), "~C")  # short, and valid in a rule
    unitless = data_units is None or ureg.Unit(data_units).dimensionless
    if units is None:
        if unitless:
            return
        raise UsageError(
            f"{da.name!r} is in {data_units!r}, so its threshold in --rule {source!r} needs "
            f"units, written after the number without spaces, e.g. "
            f"'{da.name} > 10{data_units}'"
        )
    if data_units is None:
        raise UsageError(
            f"{da.name!r} has no units attribute, so its threshold in --rule {source!r} "
            f"cannot be in {units!r}; drop the units or set the variable's units first"
        )
    if units_equal(units, data_units):
        return
    extra = (ureg.Unit(units) / ureg.Unit(data_units)).dimensionality
    if extra == {"[time]": 1}:
        fix = f"run convert-to-totals first to turn the rate {da.name!r} into amounts"
    elif extra == {"[time]": -1}:
        fix = f"write the threshold as an amount, since {da.name!r} holds amounts"
    else:
        fix = f"run unit-convert to put {da.name!r} in {units!r} first"
    raise UsageError(
        f"{da.name!r} is in {data_units!r} but its threshold in --rule {source!r} is in "
        f"{units!r}; indicator compares values as stored. Write the threshold in "
        f"{data_units!r}, or {fix}."
    )


def _strip_pint(da: xr.DataArray) -> xr.DataArray:
    if getattr(getattr(da, "pint", None), "units", None) is not None:
        return da.pint.dequantify()
    return da


def _steps(days: int, step_days: int) -> int:
    if days % step_days:
        raise UsageError(f"window {days}d is not a whole number of {step_days}-day steps")
    return days // step_days


def _evaluate_clause(da: xr.DataArray, clause, dim: str, step_days: int) -> xr.DataArray:
    da = _strip_pint(da)
    plain = clause.agg == "value" and clause.after is None and clause.within is None
    if dim not in da.dims and not plain:
        raise UsageError(
            f"{da.name!r} has no {dim!r} axis (dims {list(da.dims)}), so a window, 'after' or "
            f"'within' cannot run along it. A static field (e.g. a land-sea fraction) takes only "
            f"a plain comparison such as '{da.name} >= 0.5', which holds for every {dim}."
        )
    window = 1 if clause.agg == "value" else _steps(clause.window, step_days)
    core = _core_mask(da, clause, dim, window)
    if clause.after is not None:
        core = core.shift({dim: -_steps(clause.after, step_days)})
    if clause.within is not None:
        core = _within(core, dim, _steps(clause.within, step_days))
    if clause.negate:
        core = xr.where(core.isnull(), np.nan, 1.0 - core)
    return core.astype("float32")


def _core_mask(da: xr.DataArray, clause, dim: str, window: int) -> xr.DataArray:
    agg = clause.agg
    if agg in ("sum", "mean", "value"):
        rolled = _left_roll(da, dim, window, "sum" if agg == "value" else agg)
        compared = _OPS[clause.op](rolled, clause.threshold)
        return xr.where(rolled.isnull(), np.nan, compared.astype("float32"))
    daily = clause.daily_threshold
    if agg == "count-above":
        flag = xr.where(da.isnull(), np.nan, (da > daily).astype("float32"))
        rolled = _left_roll(flag, dim, window, "sum")
        compared = _OPS[clause.op](rolled, clause.threshold)
        return xr.where(rolled.isnull(), np.nan, compared.astype("float32"))
    if agg == "count-below":
        flag = xr.where(da.isnull(), np.nan, (da < daily).astype("float32"))
        rolled = _left_roll(flag, dim, window, "sum")
        compared = _OPS[clause.op](rolled, clause.threshold)
        return xr.where(rolled.isnull(), np.nan, compared.astype("float32"))
    if agg == "consecutive-above":
        flag = xr.where(da.isnull(), np.nan, (da > daily).astype("float32"))
        rolled = _left_roll(flag, dim, window, "sum")
        return xr.where(rolled.isnull(), np.nan, (rolled == window).astype("float32"))
    if agg == "consecutive-below":
        flag = xr.where(da.isnull(), np.nan, (da < daily).astype("float32"))
        rolled = _left_roll(flag, dim, window, "sum")
        return xr.where(rolled.isnull(), np.nan, (rolled == window).astype("float32"))
    raise UsageError(f"unsupported agg {agg!r}")


def _left_roll(da: xr.DataArray, dim: str, window: int, method: str) -> xr.DataArray:
    """Trailing rolling window, then shift labels to the left edge (forward window)."""
    if window < 1:
        raise UsageError(f"rolling window must be >= 1; got {window}")
    if window == 1:
        return da
    axis = da[dim]
    rolled = da.rolling({dim: window}, min_periods=window, center=False)
    if method == "sum":
        out = rolled.sum(skipna=False)
    elif method == "mean":
        out = rolled.mean(skipna=False)
    else:
        raise UsageError(f"unsupported rolling method {method!r}")
    out = out.isel({dim: slice(window - 1, None)})
    # Label each window by its first sample (works for any uniform step).
    out = out.assign_coords({dim: axis.values[: out.sizes[dim]]})
    return out.reindex({dim: axis})


def _within(mask: xr.DataArray, dim: str, days: int) -> xr.DataArray:
    """True if ``mask`` is True on any of the next ``days`` labels.

    One True decides it; otherwise False needs every label decided (else NaN).
    """
    if days < 1:
        raise UsageError(f"within window must be >= 1d; got {days}")
    shifted = [mask.shift({dim: -k}) for k in range(1, days + 1)]
    stack = xr.concat(shifted, dim="_look")
    complete = stack.notnull().all("_look")
    hit = (stack == 1).any("_look")
    return xr.where(hit, 1.0, xr.where(complete, 0.0, np.nan))


def _combine(a: xr.DataArray, b: xr.DataArray, how: str) -> xr.DataArray:
    a, b = xr.align(a, b, join="outer")
    if how == "and":
        both_true = (a == 1) & (b == 1)
        any_false = (a == 0) | (b == 0)
        return xr.where(both_true, 1.0, xr.where(any_false, 0.0, np.nan))
    if how == "or":
        any_true = (a == 1) | (b == 1)
        both_false = (a == 0) & (b == 0)
        return xr.where(any_true, 1.0, xr.where(both_false, 0.0, np.nan))
    raise UsageError(f"unsupported combinator {how!r}")


def _cumulative(mask: xr.DataArray, dim: str) -> xr.DataArray:
    seen = mask.fillna(0).cumsum(dim) > 0
    return xr.where(seen, 1.0, mask).astype("float32")


def _first_within(mask: xr.DataArray, dim: str, steps: int) -> xr.DataArray:
    """1 on the first True along ``dim`` and the ``steps - 1`` labels before it.

    0 on earlier labels; NaN after the first True (the sample is over), or from a
    gap on when the gap leaves the first True unknown.
    """
    seen = mask.fillna(0).cumsum(dim) > 0
    # A gap before the first True could hide it: unknown from there on, until a True.
    gap = (mask.isnull() & ~seen).cumsum(dim) > 0
    by = xr.where(seen, 1.0, xr.where(gap, np.nan, 0.0))
    before = by.shift({dim: 1}, fill_value=0.0)
    end = by.shift({dim: -(steps - 1)}, fill_value=-1.0)  # -1: the window runs past the data
    # Past the data, the first True is inside the window if it is seen by the last label.
    end = xr.where(end == -1, xr.where(by.isel({dim: -1}) == 1, 1.0, np.nan), end)
    return xr.where(before == 0, end, np.nan)


def _detect_any(mask: xr.DataArray, dim: str) -> xr.DataArray:
    hit = (mask == 1).any(dim)
    # A gap followed by decided days could hide the event: unknown, not 0. The
    # trailing NaN run is the undecidable look-ahead tail, reported on stderr.
    decided = mask.notnull()
    gap = (~decided & (decided.cumsum(dim) < decided.sum(dim))).any(dim)
    known_miss = decided.any(dim) & ~gap
    return xr.where(hit, 1.0, xr.where(known_miss, 0.0, np.nan))


def _detect_first(mask: xr.DataArray, dim: str) -> xr.Dataset:
    filled = mask.fillna(0)
    any_true = (mask == 1).any(dim)
    # A gap before the first True could hide an earlier onset: unknown, not later.
    gap_before = (mask.isnull() & (filled.cumsum(dim) == 0)).any(dim)
    known = any_true & ~gap_before
    first = filled.idxmax(dim).where(known)
    first.name = "indicator_time"
    out = first.to_dataset()
    if np.issubdtype(mask[dim].dtype, np.datetime64):
        doy = first.dt.dayofyear.astype("float32")
        out["indicator_doy"] = doy.where(known)
    return out


def _probability(mask: xr.DataArray) -> xr.DataArray:
    if "number" not in mask.dims:
        return mask.astype("float32")
    # NaN unless every member is decided: skipping undecided members biases the
    # fraction (with --cumulative, only members that already hit stay decided).
    return mask.mean("number", skipna=False).astype("float32")


def _axis(ds, time_dim: str | None) -> str | None:
    """Axis to evaluate along: a dim, a scalar coord left by selecting one step or date, or None."""
    if time_dim:
        if time_dim not in ds.dims and not (time_dim in ds.coords and ds[time_dim].ndim == 0):
            raise UsageError(f"--time-dim {time_dim!r} not in dataset dims {list(ds.dims)}")
        return time_dim
    if "time" in ds.dims and ds.sizes["time"] > 1:
        return "time"
    if "step" in ds.dims:
        return "step"
    if "time" in ds.dims:
        return "time"
    for name in ("step", "time"):
        if name in ds.coords and ds[name].ndim == 0:
            return name
    return None


def _step_days(ds, dim: str) -> int:
    """Step of ``dim`` in whole days (1 = daily, 7 = weekly)."""
    stamped = data_interval_of(ds)
    if stamped:
        days = float(parse_aggregation_period(stamped).to("day").magnitude)
    elif ds.sizes.get(dim, 0) >= 2:
        days = float(infer_timestep(ds, dim).to("day").magnitude)
    else:
        raise UsageError(
            "indicator needs data_interval (e.g. '1 day', '7 day') or at least two samples."
        )
    if days < 1 or abs(days - round(days)) > 0.05:
        raise UsageError(
            f"indicator requires a whole-day step (daily or weekly); got {days:g} day. "
            "Run aggregate-temporal --period daily or --period weekly first."
        )
    return round(days)


@weather_skill(
    name="indicator",
    version=_SKILL_VERSION,
)
@weather_skill.argument("-i", "--input", type=Dataset("any"), required=True)
@weather_skill.argument(
    "--rule",
    required=True,
    help=(
        "Named alias (icpac-onset, chc-onset) or one string of clauses joined by "
        "and/or, with thresholds in the variable's units, e.g. 'precip sum 8d >= 25mm'."
    ),
)
@weather_skill.argument("--variable", "-v", help="Override the variable named in --rule.")
@weather_skill.argument("--time-dim", help="Daily axis (default: time, else step).")
@weather_skill.argument(
    "--detect",
    choices=["first", "any"],
    help="Collapse the time axis: first True coordinate, or True anywhere.",
)
@weather_skill.argument(
    "--cumulative",
    action="store_true",
    help="Once True, stay True (has the event happened yet?).",
)
@weather_skill.argument(
    "--first-within",
    metavar="WINDOW",
    help=(
        "Window such as 7d or 1w: 1 on the first event and the WINDOW-1 dates "
        "before it, 0 on earlier dates, NaN after it."
    ),
)
@weather_skill.argument(
    "--probability",
    action="store_true",
    help="Ensemble fraction True (mean over number). No-op if there is no number dim.",
)
def indicator(
    ds, rule, variable, time_dim, detect, cumulative, first_within, probability, **kwargs
):
    """Apply a yes/no weather rule to each day or week of a Zarr, optionally reduced to an onset date or a probability."""
    spec = parse_rule(rule)
    first_days = None
    if first_within is not None:
        match = _WINDOW_RE.fullmatch(first_within.strip())
        if not match or int(match.group(1)) < 1:
            raise UsageError(f"--first-within {first_within!r} must look like '7d' or '1w'")
        first_days = int(match.group(1)) * (7 if match.group(2).lower() == "w" else 1)
    plain = all(c.agg == "value" and c.after is None and c.within is None for c in spec.clauses)
    # Windows, after/within, --detect, --cumulative and --first-within run along a
    # time axis; windows and --first-within need a whole-day step. Plain
    # comparisons alone need neither.
    dim = _axis(ds, time_dim)
    if dim is None and (not plain or detect or cumulative or first_days):
        raise UsageError(
            "could not identify a daily time or step dim; pass --time-dim. "
            f"Available dims: {list(ds.dims)}"
        )
    # One selected step or date (e.g. week 2 only): evaluate it as a length-1 axis.
    scalar_axis = dim is not None and dim not in ds.dims
    if scalar_axis:
        ds = ds.expand_dims(dim)
    step_days = 1 if plain and first_days is None else _step_days(ds, dim)
    mask = evaluate_spec(ds, spec, dim, variable, step_days)
    out = apply_reductions(
        mask,
        dim,
        cumulative=bool(cumulative),
        detect=detect,
        probability=bool(probability),
        first_within=None if first_days is None else _steps(first_days, step_days),
    )
    if scalar_axis and dim in out.dims:
        out = out.squeeze(dim)  # back to a scalar coord, as in the input
    payload = json.dumps(spec.to_json(), default=str)
    for name in out.data_vars:
        attrs = dict(out[name].attrs)
        attrs["long_name"] = f"Indicator: {spec.source}"
        if name in ("indicator", "probability"):
            attrs["units"] = "1"
        attrs["indicator_spec"] = payload
        if first_days is not None:
            # The window each label covers, so it pairs with same-length forecast windows.
            attrs["aggregation_period"] = f"{first_days} day"
            attrs["long_name"] = f"First {spec.source} in the next {first_days} days"
        out[name].attrs = attrs
    out.attrs["indicator_rule"] = spec.source
    out.attrs["indicator_expanded"] = spec.expanded
    names = ", ".join(out.data_vars)
    print(f"indicator  {spec.source}  ({names})")
    # Forward-looking rules (windows, within/after) cannot be decided at the end
    # of the series: NaN there is "not yet known", not "no event".
    ahead = max(
        max(c.window, step_days) - step_days + (c.after or c.within or 0) for c in spec.clauses
    )
    if ahead > 0:
        import datetime as _dt

        last = ds[dim].to_index()[-1]  # Timestamp, Timedelta (step), or cftime
        decided = last - _dt.timedelta(days=ahead)
        print(
            f"Note: --rule {spec.source!r} looks {ahead} day(s) past each date, so dates "
            f"after {decided} cannot be decided from data ending {last}: NaN there means "
            "'not yet known', not 'no event'. With --detect first, a missing date means "
            f"no event through {decided} or not yet known. Extend the input (later obs "
            "or a longer forecast) to decide later dates.",
            file=sys.stderr,
        )
    return out


if __name__ == "__main__":
    indicator()
