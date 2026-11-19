"""Local validation harness for the v3 forecaster (not used by main.py).

Scenarios:
  A   practice-cut style: kit p00/p01 scored against synthetic refreshed
      answers (players x6.06, earners x2.2..x3.2 by rate rank), plus
      robustness at players x3 and x8.
  A2  flat->flat control: p00/p01 scored against their own last read.
  B   long-horizon synthetic world with visible refreshes; compares the
      entry against the persistence baseline and a fixed-wide-spread
      structural ablation.
  F   format + determinism + final-scale runtime checks.

Run: python3 tests/backtest.py [kit_input_dir]
"""

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KIT_INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path.home() / "workspace/makefun/gta6-trophy-rarity-forecast-01-kit-v3"
    / "gta6-trophy-rarity-forecast-01-kit" / "input")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

spec = importlib.util.spec_from_file_location("entry_main", REPO / "main.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


# ---------------------------------------------------------------------------
# scorer (the contest's pinball loss on the logit scale)
# ---------------------------------------------------------------------------

def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    return 100.0 / (1.0 + math.exp(-x))


def pinball(forecast, answers):
    """forecast: {case_id: {trophy_id: [q x5]}}; answers: {case_id: [rates]}."""
    total, count = 0.0, 0
    for cid, rates in answers.items():
        for idx, y in enumerate(rates):
            q = forecast[cid][idx + 1]
            ly = logit_pct(y)
            for tau, qq in zip(LEVELS, q):
                d = ly - logit_pct(qq)
                total += 2.0 * max(tau * d, (tau - 1.0) * d)
                count += 1
    return total / count


def run_entry(cases):
    out = {}
    for case in cases:
        fc = entry.forecast_case(case)
        out[case["id"]] = {row["id"]: row["q"] for row in fc["trophies"]}
    return out


def persistence_forecast(cases):
    out = {}
    for case in cases:
        last = case["history"][-1]["rates"]
        spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
        out[case["id"]] = {}
        for i, trophy in enumerate(case["trophies"]):
            r = last[i] if last[i] is not None else 0.01
            lg = logit_pct(r)
            out[case["id"]][trophy["id"]] = [inv_logit(lg + z * spread) for z in Z]
    return out


def wide_ablation_forecast(cases):
    """Structural center with a FIXED wide symmetric spread (v2-style)."""
    out = {}
    for case in cases:
        snaps = entry.compress_history(case)
        stats = entry.HistoryStats(snaps, len(case["trophies"]))
        first = entry.parse_time(case["history"][0]["at"])
        target_hours = (entry.parse_time(case["target"]) - first).total_seconds() / 3600.0
        struct = entry.structural_rates(case, snaps, stats, target_hours)
        last = snaps[-1]
        out[case["id"]] = {}
        for j, trophy in enumerate(case["trophies"]):
            r = last.rates[j]
            if r is None:
                center = logit_pct(0.5)
            elif r >= 99.9:
                out[case["id"]][trophy["id"]] = [100.0] * 5
                continue
            elif struct is not None and struct[j] is not None:
                center = 0.15 * logit_pct(r) + 0.85 * logit_pct(struct[j])
            else:
                center = logit_pct(r)
            out[case["id"]][trophy["id"]] = [inv_logit(center + z * 1.1) for z in Z]
    return out


# ---------------------------------------------------------------------------
# scenario A / A2
# ---------------------------------------------------------------------------

def synth_refresh_answers(case, players_factor, base_players=638):
    last = case["history"][-1]["rates"]
    logs = [math.log(max(r, 0.01)) for r in last]
    lo, hi = min(logs), max(logs)
    ans = []
    for r, lr in zip(last, logs):
        if r >= 99.99:
            ans.append(100.0)
            continue
        frac = 0.5 if hi == lo else (lr - lo) / (hi - lo)
        g = 2.2 + (3.2 - 2.2) * frac
        earners = r / 100.0 * base_players * g
        ans.append(round(min(earners / (base_players * players_factor) * 100.0, 100.0), 2))
    return ans


# ---------------------------------------------------------------------------
# scenario B: synthetic long-horizon world
# ---------------------------------------------------------------------------

class Lcg:
    def __init__(self, seed):
        self.state = seed % (2 ** 31)

    def next(self):
        self.state = (1103515245 * self.state + 12345) % (2 ** 31)
        return self.state / (2 ** 31)


def build_scenario_b():
    launch = datetime(2026, 11, 19, tzinfo=timezone.utc)
    rng = Lcg(20261119)
    n = 55
    tiers = ["platinum"] + ["gold"] * 2 + ["silver"] * 6 + ["bronze"] * 46
    points = {"platinum": 300, "gold": 90, "silver": 30, "bronze": 15}
    trophies = [{"id": j + 1, "name": f"T{j + 1}", "description": "",
                 "tier": tiers[j], "points": points[tiers[j]], "secret": False}
                for j in range(n)]
    ceilings, speeds = [], []
    for j in range(n):
        if j == 1:
            ceilings.append(1.0); speeds.append(0.5)
        elif j == 0:
            ceilings.append(0.03); speeds.append(30.0)
        else:
            ceilings.append(0.04 + 0.85 * rng.next())
            speeds.append(2.5 + 38.0 * rng.next() * rng.next())

    def x_of(hours):
        return math.log1p(hours / 2.0)

    def players_true(hours):
        x = x_of(hours)
        return min(870.0 * (x ** 3.83), 260000.0) if x > 0 else 300.0

    def rates_true(hours):
        x = x_of(hours)
        out = []
        for j in range(n):
            if j == 1:
                out.append(100.0)
            else:
                out.append(round(100.0 * ceilings[j] * x / (x + speeds[j]), 2))
        return out

    # refresh schedule
    refresh_times = [0.0]
    t = 0.0
    while t < 200.0:
        t += 3.0 + 4.0 * rng.next()
        refresh_times.append(t)

    def displayed(hours):
        rt = max(r for r in refresh_times if r <= hours)
        return round(players_true(rt)), rates_true(rt)

    cut_hours = 2.35 * 24.0
    horizon_minutes = 6510
    target_hours = cut_hours + horizon_minutes / 60.0
    history = []
    t = 0.0
    slot = 0
    while t < cut_hours:
        p, r = displayed(t)
        at = launch + timedelta(hours=t)
        history.append({"slot": at.isoformat().replace("+00:00", "Z"),
                        "at": at.isoformat().replace("+00:00", "Z"),
                        "players_tracked": p,
                        "trophies_earned": int(sum(r) / 100.0 * p),
                        "rates": r})
        t += 0.5 if t < 24.0 else (10.0 / 60.0)
        slot += 1
    at_dt = launch + timedelta(hours=cut_hours)
    target_dt = at_dt + timedelta(minutes=horizon_minutes)
    case = {"schema": "makefun/gta6-trophies-case/v1", "id": "b00",
            "at": at_dt.isoformat().replace("+00:00", "Z"),
            "horizon_minutes": horizon_minutes,
            "target": target_dt.isoformat().replace("+00:00", "Z"),
            "levels": LEVELS, "trophies": trophies, "history": history,
            "game": {}, "notes": "synthetic"}
    _, answer = displayed(target_hours)
    return case, answer


# ---------------------------------------------------------------------------
# format / perf
# ---------------------------------------------------------------------------

def validate_output(case, payload_bytes):
    assert len(payload_bytes) <= 65536, "output too large"
    obj = json.loads(payload_bytes)
    assert set(obj.keys()) == {"trophies"}
    rows = obj["trophies"]
    assert len(rows) == len(case["trophies"])
    ids = [r["id"] for r in rows]
    assert sorted(ids) == sorted(t["id"] for t in case["trophies"])
    for r in rows:
        assert set(r.keys()) == {"id", "q"}
        q = r["q"]
        assert len(q) == 5
        assert all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 100 for v in q)
        assert all(q[i] <= q[i + 1] for i in range(4)), "not non-decreasing"
    return True


def format_perf_tests(practice_cases):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inp, out1, out2 = td / "in", td / "o1", td / "o2"
        inp.mkdir()
        for c in practice_cases:
            src = KIT_INPUT / f"{c['id']}.in"
            (inp / src.name).write_bytes(src.read_bytes())
        env = dict(**{k: v for k, v in __import__("os").environ.items()})
        env["MAKEFUN_INPUT_DIR"] = str(inp)
        for outd in (out1, out2):
            env["MAKEFUN_OUTPUT_DIR"] = str(outd)
            subprocess.run([sys.executable, "-P", "-s", str(REPO / "main.py")],
                           env=env, check=True)
        for c in practice_cases:
            b1 = (out1 / f"{c['id']}.out").read_bytes()
            b2 = (out2 / f"{c['id']}.out").read_bytes()
            assert b1 == b2, "non-deterministic output"
            validate_output(c, b1)
        print("format: p00/p01 valid, deterministic, sizes",
              {c['id']: len((out1 / f"{c['id']}.out").read_bytes()) for c in practice_cases})

    # final-scale synthetic: ~400 reads x 55 trophies
    case_b, _ = build_scenario_b()
    big = dict(case_b)
    hist = []
    t0 = datetime(2026, 11, 19, tzinfo=timezone.utc)
    last = case_b["history"][-1]
    for k in range(400):
        at = t0 + timedelta(minutes=10 * k)
        hist.append({"slot": at.isoformat().replace("+00:00", "Z"),
                     "at": at.isoformat().replace("+00:00", "Z"),
                     "players_tracked": last["players_tracked"] + k * 37,
                     "trophies_earned": last["trophies_earned"],
                     "rates": [min(99.9, r * (1 + 0.0004 * k)) if r < 99 else r
                               for r in last["rates"]]})
    big["history"] = hist
    big["id"] = "big"
    start = time.perf_counter()
    fc = entry.forecast_case(big)
    elapsed = time.perf_counter() - start
    validate_output(big, (json.dumps(fc) + "\n").encode())
    print(f"perf: final-scale case (400 reads x 55) forecast in {elapsed:.3f}s")
    return elapsed


def main():
    practice = [json.loads((KIT_INPUT / f"{cid}.in").read_text()) for cid in ("p00", "p01")]
    fc = run_entry(practice)

    for pf, label in ((6.06, "A x6.06"), (3.0, "A x3"), (8.0, "A x8")):
        answers = {c["id"]: synth_refresh_answers(c, pf) for c in practice}
        print(f"scenario {label}: entry loss = {pinball(fc, answers):.6f}")
    answers_a = {c["id"]: synth_refresh_answers(c, 6.06) for c in practice}
    print(f"scenario A persistence reference = {pinball(persistence_forecast(practice), answers_a):.6f}")
    answers_flat = {c["id"]: c["history"][-1]["rates"] for c in practice}
    print(f"scenario A2 (flat): entry loss = {pinball(fc, answers_flat):.6f}")

    case_b, answer_b = build_scenario_b()
    answers_b = {"b00": answer_b}
    print(f"scenario B: entry loss = {pinball(run_entry([case_b]), answers_b):.6f}")
    print(f"scenario B: persistence loss = {pinball(persistence_forecast([case_b]), answers_b):.6f}")
    print(f"scenario B: wide-ablation loss = {pinball(wide_ablation_forecast([case_b]), answers_b):.6f}")
    print(f"scenario B: reads={len(case_b['history'])}, last players={case_b['history'][-1]['players_tracked']}")

    format_perf_tests(practice)


if __name__ == "__main__":
    main()
