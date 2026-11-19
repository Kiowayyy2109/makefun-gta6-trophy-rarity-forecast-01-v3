"""Adaptive-spread trophy earn-rate forecaster (gta6-trophy-rarity-forecast-01, v3).

Design, in one paragraph: the displayed earn rates only move when the source
page refreshes, and a refresh moves two things at once -- the tracked player
base and each trophy's earner count. This program therefore forecasts in
earner/player space, not in raw rate space. It compresses the read history
into distinct page states, fits saturating log-time growth curves for the
player base and for each trophy's earners (trophy slopes shrunk toward their
tier mean, tier means toward the global mean), blends that structural view
with plain persistence according to how much history and horizon there is,
sizes its logit-space spread from the volatility the case itself shows, and
-- when the history shows no refresh yet -- prices the pending first refresh
as an explicit one-sided dilution mixture instead of a symmetric spread.

Run contract: `python3 -P -s /bot/main.py`, Python 3.13, standard library
only, deterministic. Outside the grader, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_OF_LEVEL = [-1.645, -0.674, 0.0, 0.674, 1.645]

RATE_LO = 0.01
RATE_HI = 99.99
# Log-time axis: x(t) = ln(1 + hours / X_SCALE). Growth of both the player
# base and earner counts is close to linear on this axis early in a launch.
X_SCALE_HOURS = 6.0
# How much of a fitted log-time slope to trust when extrapolating: the
# curves saturate, so fitted slopes are damped before use.
SLOPE_DAMP = 0.5
# Overall scale on the structural model's blend weight.
STRUCT_WEIGHT_SCALE = 0.75


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def clamp_rate(pct):
    return min(RATE_HI, max(RATE_LO, pct))


def logit_of(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def rate_of(logit_value):
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def mean(values):
    return sum(values) / len(values) if values else 0.0


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def log_time(hours):
    return math.log1p(max(hours, 0.0) / X_SCALE_HOURS)


def ols_slope(xs, ys):
    """Least-squares slope of ys on xs; 0.0 when the x-spread is negligible."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


# --------------------------------------------------------------------------
# history compression: reads -> distinct page states
# --------------------------------------------------------------------------

class Snapshot:
    __slots__ = ("hours", "players", "rates")

    def __init__(self, hours, players, rates):
        self.hours = hours      # hours since the first read
        self.players = players  # players_tracked at this state
        self.rates = rates      # per-trophy rates, nulls carried forward


def compress_history(case):
    """Collapse consecutive identical reads into distinct page states.

    Rates are carried forward over nulls (a read that lacked a trophy keeps
    the last seen value); a trophy null in every read falls back to the
    first non-null value seen anywhere, else None.
    """
    history = case["history"]
    n_trophies = len(case["trophies"])
    first_time = parse_time(history[0]["at"])
    carried = [None] * n_trophies
    snapshots = []
    for read in history:
        raw = read["rates"]
        for j in range(n_trophies):
            if raw[j] is not None:
                carried[j] = raw[j]
        hours = (parse_time(read["at"]) - first_time).total_seconds() / 3600.0
        players = read["players_tracked"]
        if snapshots:
            prev = snapshots[-1]
            if prev.players == players and prev.rates == carried:
                continue
        snapshots.append(Snapshot(hours, players, list(carried)))
    return snapshots


# --------------------------------------------------------------------------
# history statistics: cadence, step volatility, refresh jumps
# --------------------------------------------------------------------------

class HistoryStats:
    def __init__(self, snapshots, n_trophies):
        self.n_states = len(snapshots)
        gaps = [snapshots[k + 1].hours - snapshots[k].hours
                for k in range(len(snapshots) - 1)]
        self.cadence_hours = median(gaps) if gaps else None
        # Per-step logit changes, pooled and per trophy.
        self.step_diffs = [[] for _ in range(n_trophies)]
        pooled = []
        jump_ratios = []
        for k in range(len(snapshots) - 1):
            a, b = snapshots[k], snapshots[k + 1]
            if a.players and b.players and b.players != a.players:
                jump_ratios.append(math.log(b.players / a.players))
            for j in range(n_trophies):
                ra, rb = a.rates[j], b.rates[j]
                if ra is None or rb is None:
                    continue
                d = logit_of(rb) - logit_of(ra)
                self.step_diffs[j].append(d)
                pooled.append(d)
        self.pooled_diffs = pooled
        self.pooled_step_sigma = self._mad_sigma(pooled)
        self.mean_step = mean(pooled) if pooled else 0.0
        self.jump_log_ratio = median(jump_ratios) if jump_ratios else None

    @staticmethod
    def _mad_sigma(diffs):
        if not diffs:
            return 0.0
        med = median(diffs)
        mad = median([abs(d - med) for d in diffs])
        return mad / 0.6745 if mad > 0 else 0.0

    def trophy_step_sigma(self, j):
        own = self._mad_sigma(self.step_diffs[j])
        n = len(self.step_diffs[j])
        pooled = self.pooled_step_sigma
        # Shrink the trophy's own step size toward the pooled one.
        return (n * own + 4.0 * pooled) / (n + 4.0) if (n or pooled) else 0.0


# --------------------------------------------------------------------------
# structural model: saturating log-time curves for players and earners
# --------------------------------------------------------------------------

def structural_rates(case, snapshots, stats, target_hours):
    """Projected earn rate per trophy at the target, or None if unidentified.

    Players and per-trophy earners are each extrapolated on the log-time
    axis from their last observed value, using regression slopes shrunk
    hierarchically (trophy -> tier -> global). Growth is damped and capped:
    extrapolation is a guide, not a promise.
    """
    if len(snapshots) < 2:
        return None
    trophies = case["trophies"]
    n = len(trophies)
    xs = [log_time(s.hours) for s in snapshots]
    x_last = xs[-1]
    x_target = log_time(target_hours)
    dx = x_target - x_last
    if dx <= 0:
        return None

    # --- players -----------------------------------------------------------
    ln_players = [math.log(max(s.players, 1)) for s in snapshots]
    slope_p = ols_slope(xs, ln_players)
    recent = max(2, len(xs) // 2)
    slope_p_recent = ols_slope(xs[-recent:], ln_players[-recent:])
    slope_p = 0.5 * slope_p + 0.5 * slope_p_recent
    growth_p = math.exp(SLOPE_DAMP * slope_p * dx)
    growth_p = min(max(growth_p, 0.9), 12.0)
    if snapshots[-1].players and snapshots[-1].players > 0:
        # A tracked base does not meaningfully shrink between refreshes.
        growth_p = max(growth_p, 1.0)
    players_target = snapshots[-1].players * growth_p

    # --- earners per trophy -------------------------------------------------
    # Saturating curves bend downward on this axis, so the recent window's
    # slope is the safer guide to the next stretch: blend it 50/50 with the
    # full-history slope, exactly as for the player base above.
    last = snapshots[-1]
    raw_slopes = []
    for j in range(n):
        ys = []
        xs_j = []
        for s, x in zip(snapshots, xs):
            r = s.rates[j]
            if r is None or s.players <= 0:
                continue
            earners = max(r / 100.0 * s.players, 0.02)
            ys.append(math.log(earners))
            xs_j.append(x)
        if len(ys) < 2:
            raw_slopes.append(None)
            continue
        full = ols_slope(xs_j, ys)
        recent = max(2, len(ys) // 2)
        raw_slopes.append(0.5 * full + 0.5 * ols_slope(xs_j[-recent:], ys[-recent:]))

    known = [s for s in raw_slopes if s is not None]
    global_slope = mean(known) if known else 0.0
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = [raw_slopes[j] for j in range(n)
                if trophies[j].get("tier") == tier and raw_slopes[j] is not None]
        if vals:
            # Tier mean shrunk toward the global mean.
            tier_slopes[tier] = (len(vals) * mean(vals) + 3.0 * global_slope) / (len(vals) + 3.0)
        else:
            tier_slopes[tier] = global_slope

    weight = min(len(snapshots) - 1, 8)
    out = []
    for j in range(n):
        r_last = last.rates[j]
        if r_last is None:
            out.append(None)
            continue
        if r_last >= 99.9:
            out.append(100.0)
            continue
        earners_last = max(r_last / 100.0 * last.players, 0.02)
        slope = raw_slopes[j]
        tier_mean = tier_slopes.get(trophies[j].get("tier"), global_slope)
        if slope is None:
            slope = tier_mean
        else:
            slope = (weight * slope + 3.0 * tier_mean) / (weight + 3.0)
        d_ln_e = SLOPE_DAMP * slope * dx
        d_ln_e = min(max(d_ln_e, -1.0), 2.2)
        earners_target = earners_last * math.exp(d_ln_e)
        if players_target <= 0:
            out.append(r_last)
            continue
        rate = 100.0 * earners_target / players_target
        rate = min(max(rate, 0.02), 99.9)
        # Safety cap: however the curves bend, a structural rate more than
        # one logit from the last observed rate is speculation, not signal.
        lg_last = logit_of(r_last)
        lg_rate = logit_of(rate)
        if lg_rate > lg_last + 1.0:
            rate = rate_of(lg_last + 1.0)
        elif lg_rate < lg_last - 1.0:
            rate = rate_of(lg_last - 1.0)
        out.append(rate)
    return out


# --------------------------------------------------------------------------
# pending-refresh dilution mixture (history shows no refresh yet)
# --------------------------------------------------------------------------

def expected_first_refresh(case, snapshots):
    """Prior for the first refresh: (player multiple, per-trophy earners multiple).

    First principles: at an early refresh the tracked base multiplies
    several-fold (new profiles sync faster than old ones earn), while
    earners can only grow with playtime since launch, so each trophy's
    earners multiple sits in a modest band, higher for the trophies players
    earn earliest (proxied by their current rate rank).
    """
    last = snapshots[-1]
    players = max(last.players, 1)
    player_mult = 1.0 + 5.5 * math.exp(-players / 4000.0)
    rates = last.rates
    logs = []
    for r in rates:
        logs.append(math.log(max(r, 0.05)) if r is not None else None)
    known = [v for v in logs if v is not None]
    lo = min(known) if known else 0.0
    hi = max(known) if known else 0.0
    earn_mults = []
    for r, lr in zip(rates, logs):
        if r is None or r >= 99.9:
            earn_mults.append(player_mult)  # rate pinned; multiple irrelevant
            continue
        frac = 0.5 if hi <= lo or lr is None else (lr - lo) / (hi - lo)
        g = 2.0 + 1.3 * frac
        earn_mults.append(min(g, player_mult * 0.95))
    return player_mult, earn_mults


def mixture_quantiles(center, delta, weight, s_core, s_wide):
    """Quantiles of (1-w)*N(center, s_core) + w*N(center+delta, s_wide)."""
    spread = max(s_core, s_wide)
    lo_b = center + min(0.0, delta) - 7.0 * spread - 0.5
    hi_b = center + max(0.0, delta) + 7.0 * spread + 0.5
    out = []
    for tau in LEVELS:
        a, b = lo_b, hi_b
        for _ in range(50):
            mid = 0.5 * (a + b)
            cdf = ((1.0 - weight) * normal_cdf((mid - center) / s_core)
                   + weight * normal_cdf((mid - center - delta) / s_wide))
            if cdf < tau:
                a = mid
            else:
                b = mid
        out.append(rate_of(0.5 * (a + b)))
    return out


# --------------------------------------------------------------------------
# per-case forecast
# --------------------------------------------------------------------------

def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    horizon_minutes = float(case["horizon_minutes"])
    horizon_hours = horizon_minutes / 60.0
    horizon_days = horizon_minutes / 1440.0

    snapshots = compress_history(case)
    stats = HistoryStats(snapshots, n)
    last = snapshots[-1]
    checkpoint = parse_time(case["at"])
    first_time = parse_time(case["history"][0]["at"])
    target_hours = (parse_time(case["target"]) - first_time).total_seconds() / 3600.0
    del checkpoint

    struct = structural_rates(case, snapshots, stats, target_hours)

    # Blend weight for the structural view: needs distinct states to be
    # identified, and matters more the longer the horizon is.
    if struct is None:
        w_struct = 0.0
    else:
        identified = (stats.n_states - 1) / (stats.n_states + 3.0)
        horizon_pull = horizon_days / (horizon_days + 0.75)
        w_struct = min(0.9, STRUCT_WEIGHT_SCALE * (identified * (0.35 + 0.65 * horizon_pull) + 0.15 * horizon_pull))

    # Refresh cadence expectations for spread sizing.
    if stats.cadence_hours:
        steps_expected = min(max(horizon_hours / max(stats.cadence_hours, 0.5), 1.0), 16.0)
    else:
        steps_expected = 1.0

    # Spread floors: a narrow persistence-like width that grows with horizon.
    sigma_floor = 0.05 + 0.16 * math.sqrt(max(horizon_days, 0.0))
    x_last = log_time(last.hours)
    dx = max(log_time(target_hours) - x_last, 0.0)
    sigma_model = 0.35 * dx * w_struct

    flat_history = stats.n_states == 1
    player_mult, earn_mults = (None, None)
    refresh_weight = 0.0
    if flat_history:
        player_mult, earn_mults = expected_first_refresh(case, snapshots)
        # Chance the pending refresh lands before the target: grows with
        # the horizon against a few-hours refresh cadence.
        refresh_weight = min(0.8, 1.0 - math.exp(-horizon_hours / 1.9))

    rows = []
    for j, trophy in enumerate(trophies):
        r_last = last.rates[j]
        if r_last is None:
            # Never observed: fall back to a low, wide row.
            center = logit_of(0.5)
            qs = [rate_of(center + z * 0.9) for z in Z_OF_LEVEL]
            rows.append(_row(trophy["id"], qs))
            continue
        if r_last >= 99.9:
            # Earned by effectively everyone tracked; stays there.
            rows.append(_row(trophy["id"], [100.0] * 5))
            continue

        center_pers = logit_of(r_last)
        earners_last = max(r_last / 100.0 * max(last.players, 1), 0.0)

        if flat_history:
            # One-sided dilution mixture around persistence.
            mult = earn_mults[j] / player_mult
            diluted = clamp_rate(r_last * mult)
            delta = logit_of(diluted) - center_pers
            s_core = 0.08
            s_wide = 0.30 + 0.35 / math.sqrt(max(earners_last, 1.0))
            qs = mixture_quantiles(center_pers, delta, refresh_weight, s_core, s_wide)
            rows.append(_row(trophy["id"], qs))
            continue

        # Blended center: persistence vs structural, in logit space.
        if struct is not None and struct[j] is not None:
            center_struct = logit_of(struct[j])
            center = (1.0 - w_struct) * center_pers + w_struct * center_struct
        else:
            center = center_pers

        # Adaptive spread: floor + observed volatility + model distance,
        # plus small-count noise for trophies with few earners.
        vol_j = stats.trophy_step_sigma(j) * math.sqrt(steps_expected) * 0.8
        count_noise = 0.4 / math.sqrt(max(earners_last, 1.0)) * min(1.0, math.sqrt(steps_expected) / 3.0)
        sigma = math.sqrt(sigma_floor ** 2 + vol_j ** 2 + sigma_model ** 2 + count_noise ** 2)
        sigma = min(max(sigma, 0.07), 1.8)

        # Asymmetry: a dilution-dominated recent regime (rates falling on
        # net across refreshes) stretches the lower tail; the persistence
        # part of the center also leans into the expected remaining drift.
        drift_total = stats.mean_step * steps_expected
        dilution = min(max(-drift_total, 0.0), 1.2)
        growth = min(max(drift_total, 0.0), 1.2)
        lo_mult = 1.0 + 0.35 * dilution
        hi_mult = 1.0 - 0.10 * dilution + 0.15 * growth
        if growth > 0.3:
            lo_mult = 1.0 - 0.08 * growth
        shift = -0.18 * dilution * sigma * (1.0 - w_struct)

        qs = []
        for z in Z_OF_LEVEL:
            mult = lo_mult if z < 0 else hi_mult
            qs.append(rate_of(center + shift + z * sigma * mult))
        rows.append(_row(trophy["id"], qs))
    return {"trophies": rows}


def _row(trophy_id, quantiles):
    vals = []
    prev = 0.0
    for q in quantiles:
        v = round4(min(max(q, 0.0), 100.0))
        if isinstance(v, (int, float)) and v < prev:
            v = prev
        vals.append(v)
        prev = float(v)
    return {"id": trophy_id, "q": vals}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        payload = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
