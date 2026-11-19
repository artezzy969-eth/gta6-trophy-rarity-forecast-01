"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles, fitting curves at runtime from the case's own history.

Model
-----
The page's rates are step functions: Exophase refreshes its counts on its
own schedule (hours apart), so between refreshes nothing moves, and at a
refresh both the tracked-player count P and the per-trophy earner counts E
jump.  A rate is E / P, so we model the two sides separately:

* P(t): log P is fit by least squares against x = ln(1 + hours since the
  first history read).  Extrapolated to the target.
* E_i(t): log E_i, with E_i = rate_i * P / 100, is fit the same way per
  trophy, with the per-trophy slope shrunk toward the pooled (median)
  slope across trophies, because early earner counts are small integers
  and single-trophy fits are noisy.  Both fits use only the step points
  (reads where the value actually changed): between refreshes every read
  is identical, and the long flat stretches would tilt the fitted line.
  The fit is anchored at the first read (x = 0 there) when the history
  is flat, and re-anchored through the last observed point whenever a
  slope is shrunk or clamped, so a flat history still extrapolates over
  the full horizon and a fitted line never drifts from the latest level.

When the history is flat (no refresh seen yet, as on launch night) the
slopes fall back to priors taken from the refresh aggregates published in
the competition spec (638 -> 3,868 players tracked, 943 -> 4,276 trophies
earned over the first ~4.7 h): log-P slope ~1.04, log-E slope ~0.17 for
non-starter trophies.  The starter trophy (100% from the first read)
stays at 100%: every new player earns it, so E = P for it.

The median forecast is 100 * E_target / P_target, clamped to
[0.01, 99.99].  The logit spread combines a random-walk term
(0.5 * sqrt(days), as the house baseline), the fit residual, and a
refresh-jump term that grows with the expected log-P growth over the
horizon, because a refresh landing before the target moves every rate at
once.  Quantiles are inv_logit(logit(median) + z_tau * s).

Standard library only, no network, deterministic (no randomness, fixed
iteration order), well inside 8 s per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Priors used only when a history shows no movement at all (flat launch
# night).  From the spec's published first refresh: players 638 -> 3868
# and trophies earned 943 -> 4276 over ~4.67 h, i.e. ln(3868/638) /
# ln(1 + 4.67) = 1.04 for P, and for the non-starter trophies
# ln(408/305) / ln(1 + 4.67) = 0.17 for E (the starter trophy's 3230 new
# earners are its own new players, E = P there).
PRIOR_SLOPE_P = 1.04
PRIOR_SLOPE_E = 0.17

# Fitted slopes are clamped to these ranges to stop one noisy trophy
# from exploding the forecast over a 4.5-day horizon.
SLOPE_E_MIN, SLOPE_E_MAX = 0.0, 3.0
SLOPE_P_MIN, SLOPE_P_MAX = 0.0, 2.5
# Net log-rate slope (E minus P) clamp, per unit of x.
NET_SLOPE_MIN, NET_SLOPE_MAX = -2.0, 2.0

SIGMA_RW = 0.5          # random-walk logit spread per sqrt(day)
REFRESH_SIGMA = 0.30    # refresh-jump spread per sqrt(log P growth)
RESID_FLOOR = 0.15      # minimum residual spread for any trophy
RESID_CAP = 0.90


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, halves away from zero, returning int for whole
    values so the JSON stays compact."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime.  Returns None if unparsable."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def ols(xs, ys):
    """Ordinary least squares of y on x.  Returns (slope, intercept,
    resid_sd, n) or None when the fit is degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        resid = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        )
    else:
        resid = 0.0
    return slope, intercept, resid, n


def median(values):
    vals = sorted(values)
    m = len(vals)
    if m == 0:
        return 0.0
    if m % 2:
        return vals[m // 2]
    return 0.5 * (vals[m // 2 - 1] + vals[m // 2])


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0

    if not history:
        # No history at all: fall back to a flat 1% with a wide spread.
        rows = []
        for trophy in trophies:
            lg = logit(1.0)
            rows.append(
                {
                    "id": trophy["id"],
                    "q": [round4(inv_logit(lg + z * 1.2)) for z in Z],
                }
            )
        return {"trophies": rows}

    t0 = parse_time(history[0]["at"])
    if t0 is None:
        t0 = datetime(2026, 11, 19, tzinfo=timezone.utc)
    target = parse_time(case.get("target")) or t0
    # x = ln(1 + hours since the first read); x is 0 at the first read.
    xs = []
    ps = []
    for read in history:
        at = parse_time(read["at"])
        hours = 0.0 if at is None else max(0.0, (at - t0).total_seconds() / 3600.0)
        xs.append(math.log1p(hours))
        p = read.get("players_tracked")
        ps.append(float(p) if p else 0.0)
    x_target = math.log1p(max(0.0, (target - t0).total_seconds() / 3600.0))

    # --- Fit players-tracked growth -------------------------------------
    # Rates are step functions: nothing moves between Exophase refreshes,
    # so fit P on the step points only (the first read at each new P level),
    # not on every read -- the long flat stretches between refreshes would
    # otherwise tilt the fitted line and inflate the extrapolated P.
    p_step = []
    prev_p = None
    for x, p in zip(xs, ps):
        if p > 0 and p != prev_p:
            p_step.append((x, math.log(p)))
            prev_p = p
    if len(p_step) >= 2:
        fit_p = ols([x for x, _ in p_step], [y for _, y in p_step])
    else:
        fit_p = None
    if fit_p is not None:
        slope_p = min(SLOPE_P_MAX, max(SLOPE_P_MIN, fit_p[0]))
        intercept_p = fit_p[1]
    else:
        slope_p = PRIOR_SLOPE_P
        intercept_p = math.log(ps[0]) if ps and ps[0] > 0 else 0.0
    log_p_target = intercept_p + slope_p * x_target
    log_p_last = math.log(ps[-1]) if ps and ps[-1] > 0 else intercept_p

    # --- Per-trophy earner fits -----------------------------------------
    # First pass: raw per-trophy OLS fits on the step points (reads where
    # the trophy's earner count actually changed), to build the pooled slope.
    raw = []
    for i in range(len(trophies)):
        ex, ey = [], []
        prev_e = None
        for k, read in enumerate(history):
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            e = rates[i] * ps[k] / 100.0
            if e > 0 and e != prev_e:
                ex.append(xs[k])
                ey.append(math.log(e))
                prev_e = e
        distinct_e = len(set(ey))
        fit = ols(ex, ey) if distinct_e >= 2 else None
        # Last observed (x, log E): the level the forecast must stay tied to.
        last_pt = (ex[-1], ey[-1]) if ex else None
        raw.append((ex, ey, fit, distinct_e, last_pt))
    pooled_slopes = [
        min(SLOPE_E_MAX, max(SLOPE_E_MIN, f[0]))
        for _, _, f, d, _ in raw
        if f is not None and d >= 3
    ]
    pooled_slope = median(pooled_slopes) if pooled_slopes else PRIOR_SLOPE_E
    pooled_resids = [f[2] for _, _, f, d, _ in raw if f is not None and d >= 3]
    pooled_resid = median(pooled_resids) if pooled_resids else RESID_FLOOR

    # Last non-null rate per trophy (for the starter-trophy check and the
    # fallback level of trophies never observed).
    last_rates = []
    for i in range(len(trophies)):
        last = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                last = rates[i]
                break
        last_rates.append(last)

    rows = []
    for i, trophy in enumerate(trophies):
        ex, ey, fit, distinct_e, last_pt = raw[i]
        last = last_rates[i]
        if last is None:
            last = median([r for r in last_rates if r is not None] or [1.0])

        # The starter trophy: 100% from the first read.  Every new player
        # earns it, so it stays at (essentially) 100%.
        if last >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.9, 99.97, 99.99, 99.99, 99.99]})
            continue

        if fit is not None and distinct_e >= 3:
            # Shrink the noisy per-trophy slope 25% toward the pooled one,
            # then re-anchor the line through the last observed point: a
            # changed slope with the old intercept would drift the forecast
            # away from the latest observed level.
            slope_e = 0.75 * fit[0] + 0.25 * pooled_slope
            intercept_e = last_pt[1] - slope_e * last_pt[0]
            resid = fit[2]
        elif fit is not None:
            slope_e = 0.5 * fit[0] + 0.5 * pooled_slope
            intercept_e = last_pt[1] - slope_e * last_pt[0]
            resid = pooled_resid
        else:
            slope_e = pooled_slope
            # ey already holds log-earners, so the intercept is ey[0] itself;
            # only fall back to the last rate (converted to earners at the
            # first read's player count) when no earner was ever observed.
            if ey:
                intercept_e = ey[0]
            else:
                p_first = ps[0] if ps and ps[0] > 0 else 638.0
                intercept_e = math.log(max(last, 0.01) * p_first / 100.0)
            resid = pooled_resid
        slope_e = min(SLOPE_E_MAX, max(SLOPE_E_MIN, slope_e))
        # Keep the net log-rate slope inside sane bounds.  When the clamp
        # moves the slope, re-anchor the line through the last observed
        # point so the forecast stays tied to the latest level instead of
        # drifting away from the data.
        net = slope_e - slope_p
        if net > NET_SLOPE_MAX or net < NET_SLOPE_MIN:
            slope_e = slope_p + (
                NET_SLOPE_MAX if net > NET_SLOPE_MAX else NET_SLOPE_MIN
            )
            if last_pt is not None:
                intercept_e = last_pt[1] - slope_e * last_pt[0]

        log_e_target = intercept_e + slope_e * x_target
        med = 100.0 * math.exp(log_e_target - log_p_target)
        med = clamp_pct(med)

        resid = min(RESID_CAP, max(RESID_FLOOR, resid))
        spread_rw = SIGMA_RW * math.sqrt(horizon_days)
        growth = max(0.0, log_p_target - log_p_last)
        spread_refresh = REFRESH_SIGMA * math.sqrt(min(growth, 4.0))
        sigma = math.sqrt(spread_rw ** 2 + resid ** 2 + spread_refresh ** 2)

        lg = logit(med)
        qs = []
        prev = -1.0
        for z in Z:
            q = round4(inv_logit(lg + z * sigma))
            q = min(100.0, max(0.0, q))
            if q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
