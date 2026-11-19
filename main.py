"""gta6-trophy-rarity-forecast-01: a refresh-aware trophy forecaster.

Approach (see README.md for the full write-up):

* Exophase's page only moves when Exophase refreshes a game's counts.
  Between refreshes the printed rates do not move at all, so persistence
  is wrong exactly when a refresh lands -- and the practice cases straddle
  the first refresh. At a refresh the tracked-player count jumps (new
  players who have mostly not earned each trophy yet), so most rates
  FALL, more for rarer / harder trophies, less for trivial early ones.
* When the case history shows real refreshes (the final case will), fit,
  per trophy, logit(rate) against ln(1 + hours since release) over the
  distinct refresh points, weighted toward recent reads, with the slope
  shrunk toward its tier's median slope, and extrapolate to the target
  (less half a refresh gap, since the answer is the page as of Exophase's
  last refresh before the target). The spread comes from the fit's
  residuals plus an extrapolation penalty that grows with horizon.
* When the history is flat (the practice cases: every read identical),
  no trend exists to fit, so fall back to a first-refresh prior: the
  median rate is the last rate times a dilution factor (about 0.2-0.4,
  by trophy kind; a trophy at 100% stays at ~100%), with a wide spread
  that covers both "a refresh lands" and "it does not".

Standard library only, deterministic, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Prior refresh cadence (hours) for a hot new game, used only when the
# history shows no refresh at all: the launch-night page sat unchanged
# for ~4h before its first refresh.
PRIOR_REFRESH_HOURS = 5.0
# Median gap assumed between the target and Exophase's last refresh
# before it, when the history shows no refresh cadence.
PRIOR_REFRESH_GAP_HOURS = 2.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def sigmoid(y: float) -> float:
    if y >= 0:
        return 100 / (1 + math.exp(-y))
    e = math.exp(y)
    return 100 * e / (1 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_time(case: dict, history: list) -> datetime:
    rel = (case.get("game") or {}).get("release_date")
    if rel:
        try:
            dt = datetime.fromisoformat(str(rel).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    return parse_time(history[0]["at"])


def kind_of(trophy: dict) -> str:
    """Rough difficulty / earliness class from the trophy description."""
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    text = desc + " " + name
    if "chapter" in text or "final mission" in text:
        return "story"
    if "100% completion" in text or "250 stunts" in text or "gold medal" in text:
        return "grind"
    if "collectible" in text or "waink" in text or "explore all" in text \
            or "meditate" in text or "challenges" in text \
            or "garage properties" in text or "register 10" in text:
        return "collect"
    if "fishing" in text or "arcade" in text or "discover 100 vehicles" in text \
            or "cocktail" in text or "bowling" in text or "gone fishing" in text:
        return "easy"
    return "mid"


# First-refresh dilution factor (median new rate / old rate) by kind.
# Calibrated from the launch-night refresh the brief describes: players
# tracked 638 -> 3,868 (x6.06) while every rate but the 100% one fell, so
# the new cohort had earned almost nothing yet; the old cohort's own
# progress partly offsets the pure 638/3868 = 0.165 dilution.
DILUTION = {"story": 0.22, "grind": 0.18, "collect": 0.22, "easy": 0.40, "mid": 0.28}

# Soft absolute caps (percent) on a forecast median by tier: a platinum
# cannot plausibly reach 30% in week one, a bronze story trophy can.
TIER_CAP = {"platinum": 8.0, "gold": 25.0, "silver": 60.0, "bronze": 96.0}


def refresh_points(case: dict) -> list:
    """Distinct reads: collapse runs of identical (players, rates) vectors.

    Returns [(time, players, [rates...])] with None rates forward-filled
    per trophy, keeping the first time each distinct vector appeared.
    """
    trophies = case["trophies"]
    n = len(trophies)
    points = []
    last_vec = None
    filled = [None] * n
    for read in case["history"]:
        rates = list(read.get("rates") or [])
        rates = (rates + [None] * n)[:n]
        for i, r in enumerate(rates):
            if r is not None:
                filled[i] = r
        vec = (read.get("players_tracked"), tuple(filled))
        if vec == last_vec:
            continue
        last_vec = vec
        points.append((parse_time(read["at"]), read.get("players_tracked"), list(filled)))
    return points


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_var, sxx)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    dof = max(1, len(xs) - 2)
    return a, b, resid / dof, sxx


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])
    rel = release_time(case, case["history"])

    points = refresh_points(case)
    last_players = points[-1][1]
    last_rates = points[-1][2]

    # Median refresh gap observed in the history (hours), for the
    # staleness correction; fall back to the prior when unseen.
    gaps = []
    for (t0, _, _), (t1, _, _) in zip(points, points[1:]):
        gaps.append((t1 - t0).total_seconds() / 3600.0)
    gaps = [g for g in gaps if g > 0]
    med_gap = sorted(gaps)[len(gaps) // 2] if gaps else PRIOR_REFRESH_HOURS
    eff_target = target.timestamp() - 0.5 * min(med_gap, 12.0) * 3600
    eff_hours = max(0.0, (datetime.fromtimestamp(eff_target, tz=timezone.utc) - rel).total_seconds() / 3600.0)

    use_trend = len(points) >= 3
    if use_trend:
        # Time span of the distinct points in log-time; too short a span
        # means no usable trend even with 3 points.
        xs_all = [math.log1p(max(0.0, (t - rel).total_seconds() / 3600.0)) for t, _, _ in points]
        use_trend = (max(xs_all) - min(xs_all)) >= 0.3

    fits = {}  # trophy idx -> (median_logit, sigma)
    if use_trend:
        t_last = points[-1][0]
        raw_slopes = {}
        per_trophy = {}
        for i, trophy in enumerate(trophies):
            xs, ys, ws, ts = [], [], [], []
            for (t, _p, rates), x in zip(points, xs_all):
                r = rates[i]
                if r is None:
                    continue
                xs.append(x)
                ys.append(logit(r))
                age_h = max(0.0, (t_last - t).total_seconds() / 3600.0)
                ws.append(0.5 ** (age_h / 24.0))  # 24h recency half-life
                ts.append(t)
            if len(xs) >= 3 and max(xs) - min(xs) >= 0.3:
                a, b, rv, sxx = weighted_line(xs, ys, ws)
                # Recent slope: the last 36h of distinct points. A trophy's
                # rate saturates as players finish it, so the recent slope
                # (not the whole-history slope, which includes the fast
                # early climb) is what extrapolates.
                cut = t_last.timestamp() - 36 * 3600
                rx = [x for x, t in zip(xs, ts) if t.timestamp() >= cut]
                ry = [y for y, t in zip(ys, ts) if t.timestamp() >= cut]
                rw = [w for w, t in zip(ws, ts) if t.timestamp() >= cut]
                if len(rx) >= 2 and max(rx) - min(rx) >= 0.15:
                    _a2, b_r, _rv2, sxx_r = weighted_line(rx, ry, rw)
                else:
                    b_r, sxx_r, rx = b, sxx, xs
                per_trophy[i] = (a, b, rv, sxx, len(xs), xs, ys, b_r, sxx_r, len(rx))
                raw_slopes.setdefault(trophy.get("tier", "bronze"), []).append(b_r)
        tier_slope = {}
        for tier, slopes in raw_slopes.items():
            s = sorted(slopes)
            tier_slope[tier] = s[len(s) // 2] if s else 0.0
        resid_pool = {}
        for i, (a, b, rv, _sxx, _n, _xs, _ys, _br, _sx, _rn) in per_trophy.items():
            tier = trophies[i].get("tier", "bronze")
            resid_pool.setdefault(tier, []).append(rv)
        tier_resid = {}
        for tier, rvs in resid_pool.items():
            tier_resid[tier] = math.sqrt(sum(rvs) / len(rvs)) if rvs else 0.4
        for i, trophy in enumerate(trophies):
            tier = trophy.get("tier", "bronze")
            r_last = last_rates[i]
            if r_last is None:
                r_last = 0.01
            if i in per_trophy:
                a, b, rv, sxx, n, xs, ys, b_r, sxx_r, n_r = per_trophy[i]
                # Shrink the recent slope toward the tier median.
                ts = tier_slope.get(tier, 0.0)
                b_s = (n_r / (n_r + 3.0)) * b_r + (3.0 / (n_r + 3.0)) * ts
                b_s = max(-3.0, min(3.0, b_s))
                x_t = math.log1p(eff_hours)
                # Re-anchor at the last point so the forecast starts from
                # the last observed rate, then follows the shrunk recent
                # slope, damped: the further out, the more the rise
                # saturates. Cap the total logit rise/fall from the last
                # rate (story trophies may climb further than the rest).
                y_last = logit(r_last)
                x_last = xs[-1]
                dx = max(0.0, x_t - x_last)
                rise = b_s * dx / (1.0 + 0.4 * dx)
                cap_rise = 1.6 if kind_of(trophy) == "story" else 1.0
                rise = max(-1.2, min(cap_rise, rise))
                y_hat = y_last + rise
                sig_res = tier_resid.get(tier, 0.4)
                se = math.sqrt(max(rv, 0.04) * (1.0 / n_r + dx * dx / max(sxx_r, 1e-9))) if sxx_r > 0 else sig_res
                sigma = math.sqrt(sig_res ** 2 + se ** 2) + 0.15 + 0.15 * math.sqrt(horizon_d)
                sigma = min(1.05, max(0.35, sigma))
                fits[i] = (y_hat, sigma)
            else:
                fits[i] = None

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = 0.01
        tier = trophy.get("tier", "bronze")
        cap = TIER_CAP.get(tier, 96.0)
        if use_trend and fits.get(i) is not None:
            y_med, sigma = fits[i]
            if r_last >= 99.5:
                # A trophy (nearly) everyone has stays there.
                med, sigma = 99.9, 0.35
                y_med = logit(med)
            else:
                # The tier cap bounds rises only; never pull a forecast
                # below the last observed rate.
                med = min(max(cap, r_last), sigmoid(y_med))
                med = clamp(med)
                y_med = logit(med)
        elif use_trend:
            # History shows refreshes but this trophy has no usable
            # series (e.g. mostly missing rates): persist it with a
            # moderate spread, never the flat-history dilution prior.
            med = clamp(r_last)
            sigma = min(1.05, 0.60 + 0.15 * math.sqrt(horizon_d))
            y_med = logit(med)
        else:
            # Flat history: first-refresh prior (or near-persistence for
            # a very short horizon, where no refresh is expected).
            if horizon_h < 0.75:
                med = clamp(r_last * 1.02)
                sigma = 0.30 + 0.10 * math.sqrt(horizon_d)
            elif r_last >= 99.5:
                med = 99.9
                sigma = 0.35
            else:
                g = DILUTION[kind_of(trophy)]
                med = clamp(r_last * g)
                sigma = 0.90 + 0.10 * math.sqrt(horizon_d)
                sigma = min(1.30, sigma)
            y_med = logit(med)
        qs = [round4(clamp(sigmoid(y_med + z * sigma))) for z in Z]
        # Keep monotonic after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
