"""Correctness tests for indicator."""

import numpy as np
import pytest
import xarray as xr
from conftest import load_skill, make_forecast, make_gridded, run_skill, write_zarr
from weather_skills_core import UsageError
from weather_skills_core.provenance import load_history
from weather_skills_core.units import DATA_INTERVAL_ATTR


@pytest.fixture(scope="module")
def indicator_mod():
    return load_skill("indicator", "indicator")


@pytest.fixture(scope="module")
def indicator_fn(indicator_mod):
    return indicator_mod.indicator


def _daily(n_time, fill=0.0, name="precip", start="2026-01-01"):
    """Daily precipitation totals in mm."""
    ds = make_gridded(n_time=n_time, lats=(1.0,), lons=(10.0,), name=name, fill=fill, start=start)
    ds[name].attrs.update(units="mm", standard_name="lwe_thickness_of_precipitation_amount")
    ds[name].attrs[DATA_INTERVAL_ATTR] = "1 day"
    return ds


def _cell(ds, name="indicator"):
    return xr.open_zarr(ds, consolidated=True)[name].isel(latitude=0, longitude=0)


def test_parse_alias_matches_expanded(indicator_mod):
    parse_rule = indicator_mod.parse_rule
    alias = parse_rule("icpac-onset")
    written = parse_rule("precip sum 3d >= 20mm and not precip consecutive-below 1mm 7d within 21d")
    assert alias.clauses == written.clauses
    assert alias.combinator == written.combinator
    chc = parse_rule("chc-onset")
    chc_w = parse_rule("precip sum 10d > 25mm and precip sum 20d > 20mm after 10d")
    assert chc.clauses == chc_w.clauses


def test_parse_rejects_mixed_combinators(indicator_mod):
    with pytest.raises(UsageError, match="mixing"):
        indicator_mod.parse_rule(
            "precip sum 1d >= 1mm and precip sum 1d < 1mm or precip sum 1d >= 0mm"
        )


def test_parse_rejects_after_and_within(indicator_mod):
    with pytest.raises(UsageError, match="after"):
        indicator_mod.parse_rule("precip sum 3d >= 20mm after 1d within 2d")


def test_eight_day_sum_left_labeled(tmp_path, indicator_fn):
    ds = _daily(16, fill=0.0)
    ds["precip"].values[5:13, 0, 0] = 4.0  # 8 days × 4 mm = 32
    src = write_zarr(ds, tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    run_skill(indicator_fn, "-i", str(src), "-o", str(out), "--rule", "precip sum 8d >= 25mm")
    hit = _cell(out).values
    times = xr.open_zarr(out, consolidated=True).time.values
    # Left-labeled: window starting day 5 covers days 5–12.
    t5 = np.datetime64("2026-01-06")
    t4 = np.datetime64("2026-01-05")
    assert hit[times == t5] == pytest.approx(1)
    # Day 4 window includes one zero + 7×4 = 28 >= 25
    assert hit[times == t4] == pytest.approx(1)
    # Day 0 window is all zeros
    assert hit[0] == pytest.approx(0)
    assert load_history(out)[-1]["skill"] == "indicator"


def test_ten_day_sum_below(tmp_path, indicator_fn):
    wet = _daily(12, fill=1.0)
    dry = _daily(12, fill=0.5)
    wet_p = write_zarr(wet, tmp_path / "wet.zarr")
    dry_p = write_zarr(dry, tmp_path / "dry.zarr")
    run_skill(
        indicator_fn,
        "-i",
        str(wet_p),
        "-o",
        str(tmp_path / "wet_out.zarr"),
        "--rule",
        "precip sum 10d < 9mm",
    )
    run_skill(
        indicator_fn,
        "-i",
        str(dry_p),
        "-o",
        str(tmp_path / "dry_out.zarr"),
        "--rule",
        "precip sum 10d < 9mm",
    )
    assert _cell(tmp_path / "wet_out.zarr").values[0] == pytest.approx(0)
    assert _cell(tmp_path / "dry_out.zarr").values[0] == pytest.approx(1)


def test_icpac_onset_and_false_start(tmp_path, indicator_fn):
    ok = _daily(40, fill=2.0)
    ok["precip"].values[0:3, 0, 0] = 10.0
    src = write_zarr(ok, tmp_path / "ok.zarr")
    run_skill(
        indicator_fn,
        "-i",
        str(src),
        "-o",
        str(tmp_path / "ok_out.zarr"),
        "--rule",
        "icpac-onset",
    )
    assert _cell(tmp_path / "ok_out.zarr").values[0] == pytest.approx(1)

    false = _daily(40, fill=2.0)
    false["precip"].values[0:3, 0, 0] = 10.0
    false["precip"].values[3:10, 0, 0] = 0.0
    src_f = write_zarr(false, tmp_path / "false.zarr")
    run_skill(
        indicator_fn,
        "-i",
        str(src_f),
        "-o",
        str(tmp_path / "false_out.zarr"),
        "--rule",
        "icpac-onset",
    )
    assert _cell(tmp_path / "false_out.zarr").values[0] == pytest.approx(0)


def test_icpac_incomplete_lookahead_is_nan(tmp_path, indicator_fn):
    ds = _daily(10, fill=10.0)
    src = write_zarr(ds, tmp_path / "short.zarr")
    run_skill(
        indicator_fn,
        "-i",
        str(src),
        "-o",
        str(tmp_path / "short_out.zarr"),
        "--rule",
        "icpac-onset",
    )
    assert np.isnan(_cell(tmp_path / "short_out.zarr").values[0])


def test_chc_onset_strict_gt_and_follow_on(tmp_path, indicator_fn):
    ds = _daily(40, fill=0.0)
    ds["precip"].values[0:10, 0, 0] = 3.0  # 30 > 25
    ds["precip"].values[10:30, 0, 0] = 1.1  # 22 > 20
    src = write_zarr(ds, tmp_path / "chc.zarr")
    run_skill(
        indicator_fn,
        "-i",
        str(src),
        "-o",
        str(tmp_path / "chc_out.zarr"),
        "--rule",
        "chc-onset",
    )
    assert _cell(tmp_path / "chc_out.zarr").values[0] == pytest.approx(1)

    exact = _daily(40, fill=0.0)
    exact["precip"].values[0:10, 0, 0] = 2.5  # 25 not > 25
    exact["precip"].values[10:30, 0, 0] = 1.0  # 20 not > 20
    src_e = write_zarr(exact, tmp_path / "exact.zarr")
    run_skill(
        indicator_fn,
        "-i",
        str(src_e),
        "-o",
        str(tmp_path / "exact_out.zarr"),
        "--rule",
        "chc-onset",
    )
    assert _cell(tmp_path / "exact_out.zarr").values[0] == pytest.approx(0)

    dry_follow = _daily(40, fill=0.0)
    dry_follow["precip"].values[0:10, 0, 0] = 3.0
    src_d = write_zarr(dry_follow, tmp_path / "dryf.zarr")
    run_skill(
        indicator_fn,
        "-i",
        str(src_d),
        "-o",
        str(tmp_path / "dryf_out.zarr"),
        "--rule",
        "chc-onset",
    )
    assert _cell(tmp_path / "dryf_out.zarr").values[0] == pytest.approx(0)


def test_alias_matches_written_rule_on_data(tmp_path, indicator_fn):
    ds = _daily(40, fill=2.0)
    ds["precip"].values[0:3, 0, 0] = 10.0
    src = write_zarr(ds, tmp_path / "in.zarr")
    run_skill(
        indicator_fn,
        "-i",
        str(src),
        "-o",
        str(tmp_path / "alias.zarr"),
        "--rule",
        "icpac-onset",
    )
    run_skill(
        indicator_fn,
        "-i",
        str(src),
        "-o",
        str(tmp_path / "written.zarr"),
        "--rule",
        "precip sum 3d >= 20mm and not precip consecutive-below 1mm 7d within 21d",
    )
    np.testing.assert_array_equal(
        _cell(tmp_path / "alias.zarr").values,
        _cell(tmp_path / "written.zarr").values,
    )


def test_ensemble_probability(tmp_path, indicator_fn):
    ds = _daily(8, fill=0.0)
    ds = ds.expand_dims(number=[0, 1, 2, 3]).copy(deep=True)
    ds["precip"].values[0:2, :, 0, 0] = 4.0
    src = write_zarr(ds, tmp_path / "ens.zarr")
    run_skill(
        indicator_fn,
        "-i",
        str(src),
        "-o",
        str(tmp_path / "p.zarr"),
        "--rule",
        "precip sum 8d >= 25mm",
        "--probability",
    )
    out = xr.open_zarr(tmp_path / "p.zarr", consolidated=True)
    assert "probability" in out
    assert "number" not in out.dims
    assert out["probability"].isel(time=0, latitude=0, longitude=0).values == pytest.approx(0.5)


def test_detect_first_and_cumulative_probability(tmp_path, indicator_fn):
    ds = _daily(12, fill=0.0)
    ds["precip"].values[4:12, 0, 0] = 4.0
    src = write_zarr(ds, tmp_path / "in.zarr")
    run_skill(
        indicator_fn,
        "-i",
        str(src),
        "-o",
        str(tmp_path / "first.zarr"),
        "--rule",
        "precip sum 8d >= 25mm",
        "--detect",
        "first",
    )
    first = xr.open_zarr(tmp_path / "first.zarr", consolidated=True)
    assert "indicator_time" in first
    got = first["indicator_time"].isel(latitude=0, longitude=0).values
    assert np.datetime64(got, "D") == np.datetime64("2026-01-04")

    run_skill(
        indicator_fn,
        "-i",
        str(src),
        "-o",
        str(tmp_path / "cdf.zarr"),
        "--rule",
        "precip sum 8d >= 25mm",
        "--cumulative",
        "--probability",
    )
    cdf = _cell(tmp_path / "cdf.zarr", "probability").values
    assert cdf[0] == pytest.approx(0)
    assert cdf[-1] == pytest.approx(1)


def test_detect_first_rejects_probability_and_cumulative(tmp_path, indicator_fn):
    src = write_zarr(_daily(8, fill=4.0), tmp_path / "in.zarr")
    with pytest.raises(SystemExit) as exc:
        run_skill(
            indicator_fn,
            "-i",
            str(src),
            "-o",
            str(tmp_path / "out.zarr"),
            "--rule",
            "precip sum 8d >= 25mm",
            "--detect",
            "first",
            "--probability",
        )
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        run_skill(
            indicator_fn,
            "-i",
            str(src),
            "-o",
            str(tmp_path / "out2.zarr"),
            "--rule",
            "precip sum 8d >= 25mm",
            "--detect",
            "first",
            "--cumulative",
        )
    assert exc.value.code == 2


def test_sub_daily_is_refused(tmp_path, indicator_fn):
    ds = make_gridded(n_time=3, lats=(1.0,), lons=(10.0,), fill=4.0)
    ds = ds.assign_coords(
        time=np.array(["2026-01-01T00", "2026-01-01T12", "2026-01-02T00"], dtype="datetime64[ns]")
    )
    src = write_zarr(ds, tmp_path / "subdaily.zarr")
    with pytest.raises(SystemExit) as exc:
        run_skill(
            indicator_fn,
            "-i",
            str(src),
            "-o",
            str(tmp_path / "out.zarr"),
            "--rule",
            "precip sum 1d >= 1mm",
        )
    assert exc.value.code == 2


def test_missing_variable(tmp_path, indicator_fn):
    src = write_zarr(_daily(8, fill=4.0, name="tp"), tmp_path / "in.zarr")
    with pytest.raises(SystemExit) as exc:
        run_skill(
            indicator_fn,
            "-i",
            str(src),
            "-o",
            str(tmp_path / "out.zarr"),
            "--rule",
            "precip sum 8d >= 25mm",
        )
    assert exc.value.code == 2
    run_skill(
        indicator_fn,
        "-i",
        str(src),
        "-o",
        str(tmp_path / "ok.zarr"),
        "--rule",
        "precip sum 8d >= 25mm",
        "-v",
        "tp",
    )
    assert _cell(tmp_path / "ok.zarr").values[0] == pytest.approx(1)


def test_step_forecast_axis(tmp_path, indicator_fn):
    ds = make_forecast(n_step=10, lats=(1.0,), lons=(10.0,), name="precip", fill=4.0)
    ds["precip"].attrs[DATA_INTERVAL_ATTR] = "1 day"
    src = write_zarr(ds, tmp_path / "fc.zarr")
    run_skill(
        indicator_fn,
        "-i",
        str(src),
        "-o",
        str(tmp_path / "out.zarr"),
        "--rule",
        "precip sum 8d >= 25mm",
    )
    out = xr.open_zarr(tmp_path / "out.zarr", consolidated=True)
    assert "step" in out.dims
    assert out["indicator"].isel(step=0, latitude=0, longitude=0).values == pytest.approx(1)


def test_weekly_rates_probability(tmp_path, indicator_fn, capsys):
    # Weekly forecast in mm day-1: rates are compared as rates.
    ds = make_forecast(n_step=3, lats=(1.0,), lons=(10.0,), name="tp", members=4)
    ds = ds.assign_coords(step=np.arange(3) * np.timedelta64(7, "D"))
    ds["tp"].attrs.update(units="mm day-1", **{DATA_INTERVAL_ATTR: "7 day"})
    ds["tp"].values[:, :, 0, 0] = 10.0
    ds["tp"].values[:, 0, 0, 0] = [0.5, 1.0, 2.0, 3.0]
    src = write_zarr(ds, tmp_path / "fc.zarr")
    out = tmp_path / "out.zarr"
    rule = "tp mean 1w <= 1.5mm/day"
    run_skill(indicator_fn, "-i", str(src), "-o", str(out), "--rule", rule, "--probability")
    assert _cell(out, "probability").values.tolist() == pytest.approx([0.5, 0.0, 0.0])
    for bad in ("tp mean 10d <= 9mm/day", "tp sum 1w <= 10mm"):
        with pytest.raises(SystemExit):
            run_skill(indicator_fn, "-i", str(src), "-o", str(tmp_path / "bad.zarr"), "--rule", bad)
    assert "convert-to-totals" in capsys.readouterr().err


@pytest.mark.parametrize(
    "units, rule, hint",
    [
        ("mm day-1", "precip sum 3d >= 20mm", "run convert-to-totals"),
        ("mm", "precip >= 5mm/day", "write the threshold as an amount"),
        ("kg m-2 s-1", "precip >= 5mm/day", "run unit-convert"),
        ("mm", "precip >= 5", "needs units, written after the number"),
    ],
)
def test_threshold_units_must_match_the_data(tmp_path, indicator_fn, capsys, units, rule, hint):
    ds = _daily(8, fill=10.0)
    ds["precip"].attrs["units"] = units
    src = write_zarr(ds, tmp_path / "in.zarr")
    with pytest.raises(SystemExit) as exc:
        run_skill(indicator_fn, "-i", str(src), "-o", str(tmp_path / "o.zarr"), "--rule", rule)
    assert exc.value.code == 2
    assert hint in capsys.readouterr().err


def test_threshold_units_match_whatever_the_spelling(tmp_path, indicator_fn):
    ds = _daily(3)
    ds["precip"].values[:, 0, 0] = [2.0, 6.0, 9.0]
    ds["precip"].attrs["units"] = "mm day-1"
    src = write_zarr(ds, tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    run_skill(indicator_fn, "-i", str(src), "-o", str(out), "--rule", "precip >= 5mm/day")
    assert _cell(out).values.tolist() == [0.0, 1.0, 1.0]


def test_temperature_is_compared_as_stored(tmp_path, indicator_fn, capsys):
    ds = _daily(4, name="t2m")
    ds["t2m"].values[:, 0, 0] = [300.0, 310.0, 300.0, 310.0]
    ds["t2m"].attrs.update(units="K", standard_name="air_temperature")
    src = write_zarr(ds, tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    run_skill(indicator_fn, "-i", str(src), "-o", str(out), "--rule", "t2m mean 1d >= 308.15K")
    assert _cell(out).values.tolist() == [0.0, 1.0, 0.0, 1.0]
    with pytest.raises(SystemExit):
        run_skill(
            indicator_fn, "-i", str(src), "-o", str(tmp_path / "c.zarr"), "--rule", "t2m > 35degC"
        )
    assert "run unit-convert" in capsys.readouterr().err


@pytest.mark.parametrize(
    "rule, hint",
    [
        ("precip count-above 1mm 3d >= 2mm", "counts steps"),
        ("precip > 5parsecs_per_fortnight", "unknown units"),
        ("precip > 5 mm", "unexpected 'mm'"),
    ],
)
def test_threshold_units_parse_errors(indicator_mod, rule, hint):
    with pytest.raises(UsageError, match=hint):
        indicator_mod.parse_rule(rule)


def test_detect_first_is_unknown_after_a_gap(tmp_path, indicator_fn):
    """Missing days before the first observed onset could hide an earlier one."""
    ds = _daily(10)
    ds["precip"].values[:, 0, 0] = [np.nan, np.nan, 0, 0, 10, 10, 10, 0, 0, 0]
    src = write_zarr(ds, tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    run_skill(
        indicator_fn,
        "-i",
        str(src),
        "-o",
        str(out),
        "--rule",
        "precip sum 3d >= 20mm",
        "--detect",
        "first",
    )
    assert np.isnat(_cell(out, "indicator_time").values)


def test_lookahead_tail_is_reported(tmp_path, indicator_fn, capsys):
    """icpac-onset looks 27 days ahead: say which dates cannot be decided yet."""
    src = write_zarr(_daily(40, fill=10.0), tmp_path / "in.zarr")
    run_skill(
        indicator_fn, "-i", str(src), "-o", str(tmp_path / "out.zarr"),
        "--rule", "icpac-onset", "--detect", "first",
    )  # fmt: skip
    err = capsys.readouterr().err
    assert "looks 27 day(s) past each date" in err
    assert "after 2026-01-13" in err  # data end 2026-02-09 minus 27 days


def test_backward_free_rule_has_no_tail_note(tmp_path, indicator_fn, capsys):
    src = write_zarr(_daily(10, fill=10.0), tmp_path / "in.zarr")
    run_skill(
        indicator_fn, "-i", str(src), "-o", str(tmp_path / "out.zarr"),
        "--rule", "precip sum 1d >= 5mm",
    )  # fmt: skip
    assert "looks" not in capsys.readouterr().err


def test_nan_day_makes_every_covering_window_unknown(tmp_path, indicator_fn):
    ds = _daily(8)
    ds["precip"].values[:, 0, 0] = [0, 0, np.nan, 0, 0, 0, 0, 0]
    src = write_zarr(ds, tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    run_skill(
        indicator_fn, "-i", str(src), "-o", str(out), "--rule", "precip count-below 1mm 3d >= 3"
    )
    got = _cell(out).values  # windows starting days 0-2 cover the gap; days 6-7 are the tail
    np.testing.assert_array_equal(got, [np.nan, np.nan, np.nan, 1, 1, 1, np.nan, np.nan])


def test_cumulative_probability_tail_is_unknown(tmp_path, indicator_fn):
    """A member that has not hit by the tail is undecided, so P is unknown, not 1."""
    ds = _daily(6).expand_dims(number=[0, 1]).copy(deep=True)
    ds["precip"].values[0, :, 0, 0] = [0, 30, 0, 0, 0, 0]  # member 0 onset on day 0
    src = write_zarr(ds, tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    run_skill(
        indicator_fn, "-i", str(src), "-o", str(out),
        "--rule", "precip sum 3d >= 20mm", "--cumulative", "--probability",
    )  # fmt: skip
    np.testing.assert_array_equal(
        _cell(out, "probability").values, [0.5, 0.5, 0.5, 0.5, np.nan, np.nan]
    )


def test_probability_is_unknown_when_a_member_is_missing(tmp_path, indicator_fn):
    ds = _daily(2).expand_dims(number=[0, 1, 2]).copy(deep=True)
    ds["precip"].values[:, 0, 0, 0] = [30, np.nan, 0]
    src = write_zarr(ds, tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    run_skill(
        indicator_fn, "-i", str(src), "-o", str(out),
        "--rule", "precip sum 1d >= 20mm", "--probability",
    )  # fmt: skip
    np.testing.assert_array_equal(_cell(out, "probability").values, [np.nan, 0.0])


def test_detect_any_is_unknown_across_a_gap(tmp_path, indicator_fn):
    """A missing day is not a dry day: no hit with a gap is unknown, not 0."""
    gap = _daily(5)
    gap["precip"].values[:, 0, 0] = [0, np.nan, 0, 0, 0]
    tail = _daily(5)
    tail["precip"].values[:, 0, 0] = [0, 0, 0, 0, np.nan]  # trailing run = undecided tail
    for ds, want in ((gap, np.nan), (tail, 0.0)):
        src = write_zarr(ds, tmp_path / "in.zarr")
        out = tmp_path / "out.zarr"
        run_skill(
            indicator_fn, "-i", str(src), "-o", str(out),
            "--rule", "precip sum 1d >= 20mm", "--detect", "any",
        )  # fmt: skip
        np.testing.assert_array_equal(_cell(out).values, want)


def test_within_one_true_day_decides_an_incomplete_lookahead(tmp_path, indicator_fn):
    """A dry spell already seen is a known false start even if later days are undecided."""
    ds = _daily(7)
    ds["precip"].values[:, 0, 0] = [0, 0, 0, 0, 0, 5, 5]
    src = write_zarr(ds, tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    run_skill(
        indicator_fn, "-i", str(src), "-o", str(out),
        "--rule", "not precip consecutive-below 1mm 3d within 5d",
    )  # fmt: skip
    # Spell mask is 1,1,1,0,0,nan,nan: day 0/1 see a spell; day 2 needs days 5-7.
    np.testing.assert_array_equal(_cell(out).values, [0, 0] + [np.nan] * 5)


def _first_within(tmp_path, indicator_fn, values, window, *extra):
    ds = _daily(len(values)).copy(deep=True)
    ds["precip"].values[:, 0, 0] = values
    src = write_zarr(ds, tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    run_skill(
        indicator_fn, "-i", str(src), "-o", str(out),
        "--rule", "precip sum 1d >= 10mm", "--first-within", window, *extra,
    )  # fmt: skip
    return out


def test_first_within_blanks_dates_after_the_event(tmp_path, indicator_fn):
    """1 while the first event is inside the window, NaN once it already happened."""
    rain = [0, 0, 0, 0, 0, 20, 0, 0, 0, 20, 0, 0]
    out = _first_within(tmp_path, indicator_fn, rain, "3d")
    expect = [0, 0, 0, 1, 1, 1] + [np.nan] * 6
    np.testing.assert_array_equal(_cell(out).values, expect)
    attrs = _cell(out).attrs
    assert attrs["aggregation_period"] == "3 day"
    assert attrs["units"] == "1"


def test_first_within_gap_and_tail_are_unknown(tmp_path, indicator_fn):
    """A gap before the event hides the first date; with no event, the tail is undecided."""
    gap = [0, 0, np.nan, 0, 0, 20, 0, 0]
    np.testing.assert_array_equal(
        _cell(_first_within(tmp_path, indicator_fn, gap, "2d")).values, [0] + [np.nan] * 7
    )
    dry = [0, 0, 0, 0, 0]
    out = _first_within(tmp_path / "dry", indicator_fn, dry, "1w")
    np.testing.assert_array_equal(_cell(out).values, [np.nan] * 5)
    out = _first_within(tmp_path / "dry2", indicator_fn, dry, "2d")
    np.testing.assert_array_equal(_cell(out).values, [0, 0, 0, 0, np.nan])


@pytest.mark.parametrize(
    "rain, window, expect",
    [
        ([0, 0, 0, 0, 20], "3d", [0, 0, 1, 1, 1]),
        ([0, 0, 0, 20, 0], "3d", [0, 1, 1, 1, np.nan]),
        ([0, 0, np.nan, 0, 20], "3d", [np.nan, np.nan, 1, np.nan, np.nan]),
    ],
)
def test_first_within_event_near_the_end(tmp_path, indicator_fn, rain, window, expect):
    """A window running past the data is decided when the first event is already in it."""
    out = _first_within(tmp_path, indicator_fn, rain, window)
    np.testing.assert_array_equal(_cell(out).values, expect)


def test_first_within_counts_in_steps_of_the_data(tmp_path, indicator_fn):
    ds = make_forecast(n_step=4, lats=(1.0,), lons=(10.0,), name="tp", fill=0.0)
    ds = ds.assign_coords(step=np.arange(4) * np.timedelta64(7, "D"))
    ds["tp"].attrs[DATA_INTERVAL_ATTR] = "7 day"
    ds["tp"].values[2, 0, 0] = 20.0
    src = write_zarr(ds, tmp_path / "wk.zarr")
    out = tmp_path / "out.zarr"
    run_skill(
        indicator_fn, "-i", str(src), "-o", str(out), "--rule", "tp > 10mm", "--first-within", "2w"
    )
    np.testing.assert_array_equal(_cell(out).values, [0, 1, 1, np.nan])


def test_first_within_needs_a_time_axis(tmp_path, indicator_fn):
    lsm = xr.Dataset(
        {"land_sea_mask": (("latitude", "longitude"), [[0.2, 0.7]], {"units": "1"})},
        coords={"latitude": [1.0], "longitude": [10.0, 11.0]},
    )
    src = write_zarr(lsm, tmp_path / "lsm.zarr")
    with pytest.raises(SystemExit) as exc:
        run_skill(
            indicator_fn, "-i", str(src), "-o", str(tmp_path / "o.zarr"),
            "--rule", "land_sea_mask >= 0.5", "--first-within", "7d",
        )  # fmt: skip
    assert exc.value.code == 2


@pytest.mark.parametrize("extra", [["--cumulative"], ["--probability"], ["--detect", "any"]])
def test_first_within_rejects_other_reductions(tmp_path, indicator_fn, extra):
    with pytest.raises(SystemExit) as exc:
        _first_within(tmp_path, indicator_fn, [0, 20, 0], "2d", *extra)
    assert exc.value.code == 2


def test_plain_threshold_parses_as_one_step_value(indicator_mod):
    spec = indicator_mod.parse_rule("tp > 25mm")
    (clause,) = spec.clauses
    assert (clause.variable, clause.agg, clause.op, clause.threshold) == ("tp", "value", ">", 25.0)
    clause = indicator_mod.parse_rule("not precip <= 1mm within 3d").clauses[0]
    assert clause.negate and clause.agg == "value" and clause.within == 3


def test_plain_threshold_weekly_probability_matches_one_week_mean(tmp_path, indicator_fn):
    # Weekly forecast in mm day-1: 'tp > 3.5mm/day' compares each week's own rate.
    ds = make_forecast(n_step=2, lats=(1.0,), lons=(10.0,), name="tp", members=4)
    ds = ds.assign_coords(step=np.arange(2) * np.timedelta64(7, "D"))
    ds["tp"].attrs.update(units="mm day-1", **{DATA_INTERVAL_ATTR: "7 day"})
    ds["tp"].values[:, :, 0, 0] = 10.0
    ds["tp"].values[:, 0, 0, 0] = [1.0, 3.0, 4.0, 5.0]
    src = write_zarr(ds, tmp_path / "fc.zarr")
    plain, summed = tmp_path / "plain.zarr", tmp_path / "mean.zarr"
    for out, rule in ((plain, "tp > 3.5mm/day"), (summed, "tp mean 1w > 3.5mm/day")):
        run_skill(indicator_fn, "-i", str(src), "-o", str(out), "--rule", rule, "--probability")
    assert _cell(plain, "probability").values.tolist() == pytest.approx([0.5, 1.0])
    np.testing.assert_array_equal(
        _cell(plain, "probability").values, _cell(summed, "probability").values
    )


def test_plain_threshold_daily_has_no_lookahead_note(tmp_path, indicator_fn, capsys):
    ds = _daily(5, fill=0.0)
    ds["precip"].values[2, 0, 0] = 30.0
    src = write_zarr(ds, tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    run_skill(indicator_fn, "-i", str(src), "-o", str(out), "--rule", "precip >= 30mm")
    assert _cell(out).values.tolist() == pytest.approx([0, 0, 1, 0, 0])
    assert "cannot be decided" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "rule, hint",
    [
        ("list", "Examples:"),
        ("precip consecutive-below 1mm 10d >= 7", "take no comparator"),
        ("precip sum 3d >= 20mm then precip sum 1d >= 1mm", "join with 'and' or 'or'"),
    ],
)
def test_parse_errors_show_examples(indicator_mod, rule, hint):
    with pytest.raises(UsageError) as exc:
        indicator_mod.parse_rule(rule)
    assert hint in str(exc.value)
    assert "'precip sum 8d >= 25mm'" in str(exc.value)


@pytest.mark.parametrize("rule", ["tp > 25mm > 3", "precip sum 3d >= 20mm >= 5"])
def test_extra_comparator_hint_only_for_consecutive(indicator_mod, rule):
    with pytest.raises(UsageError) as exc:
        indicator_mod.parse_rule(rule)
    assert "unexpected '>" in str(exc.value)
    assert "take no comparator" not in str(exc.value)


def _weekly_forecast():
    ds = make_forecast(n_step=3, lats=(1.0,), lons=(10.0,), name="tp", members=4)
    ds = ds.assign_coords(step=np.arange(3) * np.timedelta64(7, "D"))
    ds["tp"].attrs.update(units="mm day-1", **{DATA_INTERVAL_ATTR: "7 day"})
    ds["tp"].values[:, 1, 0, 0] = [1.0, 3.0, 4.0, 5.0]
    return ds


def test_single_selected_week_probability(tmp_path, indicator_fn):
    # 'select --dim step' leaves step as a scalar coord; one week is still a valid axis.
    src = write_zarr(_weekly_forecast().isel(step=1), tmp_path / "wk2.zarr")
    out = tmp_path / "out.zarr"
    rule = "tp > 3.5mm/day"
    run_skill(indicator_fn, "-i", str(src), "-o", str(out), "--rule", rule, "--probability")
    prob = _cell(out, "probability")
    assert "step" not in prob.dims
    assert prob.step.values == np.timedelta64(7, "D")
    assert float(prob) == pytest.approx(0.5)


def test_single_selected_week_needs_data_interval(tmp_path, indicator_fn, capsys):
    ds = _weekly_forecast().isel(step=1)
    del ds["tp"].attrs[DATA_INTERVAL_ATTR]
    src = write_zarr(ds, tmp_path / "wk2.zarr")
    with pytest.raises(SystemExit) as exc:
        rule = "tp mean 1w > 3.5mm/day"
        run_skill(indicator_fn, "-i", str(src), "-o", str(tmp_path / "o.zarr"), "--rule", rule)
    assert exc.value.code == 2
    assert "data_interval" in capsys.readouterr().err


def test_static_field_plain_comparison(tmp_path, indicator_fn):
    lsm = xr.Dataset(
        {"land_sea_mask": (("latitude", "longitude"), [[0.2, 0.7]], {"units": "1"})},
        coords={"latitude": [1.0], "longitude": [10.0, 11.0]},
    )
    src = write_zarr(lsm, tmp_path / "lsm.zarr")
    out = tmp_path / "land.zarr"
    run_skill(indicator_fn, "-i", str(src), "-o", str(out), "--rule", "land_sea_mask >= 0.5")
    mask = xr.open_zarr(out, consolidated=True)["indicator"]
    assert mask.dims == ("latitude", "longitude")
    np.testing.assert_array_equal(mask.values, [[0.0, 1.0]])


def test_static_variable_next_to_daily_series(tmp_path, indicator_fn):
    ds = _daily(3, fill=2.0)
    ds["land_sea_mask"] = (("latitude", "longitude"), [[0.9]], {"units": "1"})
    src = write_zarr(ds, tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    rule = "land_sea_mask >= 0.5 and precip >= 1mm"
    run_skill(indicator_fn, "-i", str(src), "-o", str(out), "--rule", rule, "--time-dim", "time")
    np.testing.assert_array_equal(_cell(out).values, [1.0, 1.0, 1.0])
    assert xr.open_zarr(out, consolidated=True)["indicator"].dims == ds["precip"].dims
    with pytest.raises(SystemExit) as exc:
        run_skill(
            indicator_fn,
            "-i",
            str(src),
            "-o",
            str(tmp_path / "bad.zarr"),
            "--rule",
            "land_sea_mask sum 2d >= 1 and precip >= 1mm",
        )
    assert exc.value.code == 2


@pytest.mark.parametrize("extra", [["--detect", "any"], ["--cumulative"], ["--first-within", "2d"]])
def test_static_only_rule_with_time_reduction_exits_2(tmp_path, indicator_fn, extra, capsys):
    ds = _daily(3, fill=2.0)
    ds["land_sea_mask"] = (("latitude", "longitude"), [[0.9]], {"units": "1"})
    src = write_zarr(ds, tmp_path / "in.zarr")
    with pytest.raises(SystemExit) as exc:
        run_skill(indicator_fn, "-i", str(src), "-o", str(tmp_path / "out.zarr"),
                  "--rule", "land_sea_mask >= 0.5", *extra)  # fmt: skip
    assert exc.value.code == 2
    assert "without a 'time' axis" in capsys.readouterr().err


def test_hourly_plain_comparison_with_detect_any(tmp_path, indicator_fn):
    ds = make_gridded(n_time=6, lats=(1.0,), lons=(10.0,), name="t2m", fill=20.0)
    ds = ds.assign_coords(time=ds.time[0].values + np.arange(6) * np.timedelta64(1, "h"))
    ds["t2m"].attrs = {"units": "degree_Celsius", "standard_name": "air_temperature"}
    ds["t2m"][3] = 36.0
    src = write_zarr(ds, tmp_path / "hourly.zarr")
    out = tmp_path / "out.zarr"
    run_skill(
        indicator_fn, "-i", str(src), "-o", str(out), "--rule", "t2m > 35degC", "--detect", "any"
    )
    assert float(_cell(out)) == 1.0


def test_static_precip_rate_is_compared_as_stored(tmp_path, indicator_fn):
    pr = xr.Dataset(
        {"precip": (("latitude", "longitude"), [[1.0, 2.0]], {"units": "mm day-1"})},
        coords={"latitude": [1.0], "longitude": [10.0, 11.0]},
    )
    src = write_zarr(pr, tmp_path / "pr.zarr")
    out = tmp_path / "out.zarr"
    run_skill(indicator_fn, "-i", str(src), "-o", str(out), "--rule", "precip > 1.5mm/day")
    np.testing.assert_array_equal(
        xr.open_zarr(out, consolidated=True)["indicator"].values, [[0.0, 1.0]]
    )


def test_output_keeps_input_dim_order(tmp_path, indicator_fn):
    ds = make_forecast(n_step=3, lats=(1.0,), lons=(10.0, 11.0), name="tp", fill=5.0, members=2)
    ds["tp"].attrs[DATA_INTERVAL_ATTR] = "1 day"
    ds["land_sea_mask"] = (("latitude", "longitude"), [[0.2, 0.9]], {"units": "1"})
    src = write_zarr(ds, tmp_path / "in.zarr")
    out = tmp_path / "out.zarr"
    run_skill(
        indicator_fn, "-i", str(src), "-o", str(out), "--rule", "land_sea_mask >= 0.5 and tp > 1mm"
    )
    mask = xr.open_zarr(out, consolidated=True)["indicator"]
    assert mask.dims == ("number", "step", "latitude", "longitude")
    np.testing.assert_array_equal(mask.isel(number=0, step=0).values, [[0.0, 1.0]])
