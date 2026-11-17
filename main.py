#!/usr/bin/env python3
"""GTA VI Twitch launch-day forecast (gta6-twitch-ccu-forecast-01).

Forecasts hours 2..24 of launch-day Twitch viewers as five quantiles each,
from the hour-1 print, using a per-hour log-ratio ridge regression on a
clean-filtered set of comparable launches.

Model (log ratios to the hour-1 print p1):
  * anchor p1 from the known hour-1 row (with fallbacks);
  * build each comparable's release-day curve (hours 1..24);
  * clean-set filter: drop early-access-contaminated, weak and target-history
    tiers, and any comparable whose hour-1 print is < 3% of its own
    release-day peak (simultaneous-global-unlock shape);
  * per hour h = 2..24, ridge line of ln(viewers_h / hour1) on ln(hour1),
    penalty lambda = 0.1 on the slope only;
  * evaluate the line at most at the largest clean hour-1 (extrapolation
    clamp), then add a kernel-weighted (bandwidth 1.0 in log units) median
    residual correction;
  * per-hour local sigma: kernel-weighted residual spread, smoothed with a
    centred (1,2,1)/4 moving average, then scaled by 0.8, floored at 0.08
    and capped at 0.9.

Stdlib only. Deterministic: pure function of the input case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = {
    0.05: -1.6448536269514722,
    0.25: -0.6744897501960817,
    0.5: 0.0,
    0.75: 0.6744897501960817,
    0.95: 1.6448536269514722,
}
LAMBDA = 0.1
BANDWIDTH = 1.0
SIGMA_SCALE = 0.8
SIGMA_FLOOR = 0.08
SIGMA_CAP = 0.9
BASELINE_SD = 0.15
PEAK_RATIO_MIN = 0.03
MIN_POINTS = 5

# Diagnostics from the most recent forecast_case() call (for validation).
LAST_STATS = {}


def _parse_time(text):
    """Parse an ISO-8601 timestamp (UTC) into an aware datetime."""
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _anchor_p1(case):
    """Hour-1 print, with fallbacks down the history series."""
    known = case.get("known") or []
    row = None
    for r in known:
        if isinstance(r, dict) and r.get("hour") == 1:
            row = r
            break
    if row is None and known:
        row = known[0] if isinstance(known[0], dict) else None
    if row is not None:
        v = row.get("viewers")
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    for key in ("history_hourly", "history"):
        series = case.get(key) or []
        for pt in reversed(series):
            if isinstance(pt, dict):
                v = pt.get("viewers")
                if isinstance(v, (int, float)) and v > 0:
                    return float(v)
    return 1000.0


def _day_curve(comp):
    """Map release-day hour (1..24) -> viewers, first point per hour wins."""
    start = _parse_time(comp.get("day_start"))
    if start is None:
        return {}
    curve = {}
    for pt in comp.get("hourly") or []:
        if not isinstance(pt, dict):
            continue
        v = pt.get("viewers")
        if v is None or not isinstance(v, (int, float)) or v <= 0:
            continue
        t = _parse_time(pt.get("from"))
        if t is None:
            continue
        h = int((t - start).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and h not in curve:
            curve[h] = float(v)
    return curve


def _clean_comparables(case):
    """Usable comparables that pass the clean-set filter, sorted by name.

    Returns (clean, usable_count, dropped_counts) where dropped_counts maps
    a reason to how many comparables it removed.
    """
    usable = []
    for comp in case.get("comparables") or []:
        if not isinstance(comp, dict):
            continue
        curve = _day_curve(comp)
        if 1 not in curve:
            continue  # no hour-1 point: unusable (also drops target-history)
        usable.append((comp.get("name") or "", comp, curve))
    usable.sort(key=lambda item: item[0])

    clean = []
    dropped = {"tier": 0, "peak_ratio": 0}
    for _name, comp, curve in usable:
        unlock = comp.get("unlock") or ""
        if not isinstance(unlock, str):
            unlock = str(unlock)
        if "Tier: early-access-contaminated" in unlock:
            dropped["tier"] += 1
            continue
        if "Tier: weak" in unlock:
            dropped["tier"] += 1
            continue
        if "Tier: target-history" in unlock:
            dropped["tier"] += 1
            continue
        peak = max(curve[h] for h in curve if 1 <= h <= 24)
        if peak <= 0 or curve[1] / peak < PEAK_RATIO_MIN:
            dropped["peak_ratio"] += 1
            continue
        clean.append((_name, comp, curve))
    return clean, len(usable), dropped


def _weighted_median(pairs):
    """Weighted median of residuals: pairs are (residual, weight).

    If every weight underflows to zero (target far outside the training
    range in log units), fall back to the plain unweighted median so the
    correction stays defined instead of silently returning 0.
    """
    ordered = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _r, w in ordered)
    if total <= 0:
        n = len(ordered)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return ordered[mid][0]
        return (ordered[mid - 1][0] + ordered[mid][0]) / 2.0
    acc = 0.0
    for r, w in ordered:
        acc += w
        if acc >= total / 2.0:
            return r
    return ordered[-1][0]


def _baseline_row(p1, h):
    """House baseline quantiles for hour h (center 0, sd grows with time)."""
    sd = BASELINE_SD * math.sqrt(h - 1)
    return [p1 * math.exp(sd * Z[tau]) for tau in LEVELS]


def _round_half_up(value):
    """Round to 2 decimals, half-up, via Decimal on the decimal string."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _post_process(qs):
    """Enforce non-decreasing, clamp, round half-up to 2 dp, re-enforce."""
    out = list(qs)
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    out = [min(max(v, 0.01), 1e9) for v in out]
    out = [_round_half_up(v) for v in out]
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case):
    """Forecast one case -> {"hours": [{"hour": h, "q": [...]}, ...]}."""
    p1 = _anchor_p1(case)
    log_p1 = math.log(p1)
    clean, usable_count, dropped = _clean_comparables(case)

    # First pass per hour: fitted centre / raw sigma, or baseline fallback.
    centres = {}
    sd_raw = {}
    is_baseline = {}
    for h in range(2, 25):
        pts = []
        for _name, _comp, curve in clean:
            if h in curve:
                x = math.log(curve[1])
                y = math.log(curve[h] / curve[1])
                pts.append((x, y))
        if len(pts) < MIN_POINTS:
            centres[h] = 0.0
            sd_raw[h] = BASELINE_SD * math.sqrt(h - 1)
            is_baseline[h] = True
            continue
        is_baseline[h] = False
        n = len(pts)
        mx = sum(x for x, _y in pts) / n
        my = sum(y for _x, y in pts) / n
        sxx = sum((x - mx) ** 2 for x, _y in pts)
        sxy = sum((x - mx) * (y - my) for x, y in pts)
        b = sxy / (sxx + LAMBDA)
        a = my - b * mx
        x_max = max(x for x, _y in pts)
        x_eval = min(log_p1, x_max)
        centre_fit = a + b * x_eval
        # Kernel weights around the target's log hour-1.
        weighted = []
        for x, y in pts:
            w = math.exp(-0.5 * ((x - log_p1) / BANDWIDTH) ** 2)
            weighted.append((y - (a + b * x), w))
        # If the kernel collapses onto effectively one point (target far
        # outside the training range, weights underflowed), the weighted
        # spread degenerates toward zero. Fall back to uniform weights so
        # the correction and sigma stay honest; in-range targets are
        # unaffected (their effective sample size is well above this).
        w_sum0 = sum(w for _r, w in weighted)
        w_sq0 = sum(w * w for _r, w in weighted)
        ess = (w_sum0 * w_sum0 / w_sq0) if w_sq0 > 0 else 0.0
        if ess < 1.5:
            weighted = [(r, 1.0) for r, _w in weighted]
        correction = _weighted_median(weighted)
        centres[h] = centre_fit + correction
        w_sum = sum(w for _r, w in weighted)
        if w_sum > 0:
            var = sum(w * (r - correction) ** 2 for r, w in weighted) / w_sum
            sd_raw[h] = math.sqrt(max(var, 0.0))
        else:
            # All kernel weights underflowed: unweighted spread instead.
            n = len(weighted)
            mean_r = sum(r for r, _w in weighted) / n
            var = sum((r - mean_r) ** 2 for r, _w in weighted) / n
            sd_raw[h] = math.sqrt(max(var, 0.0))

    # Second pass: smooth sd_raw across hours, apply guards, build rows.
    rows = []
    sd_used = {}
    for h in range(2, 25):
        if is_baseline[h]:
            qs = _baseline_row(p1, h)
            sd_used[h] = BASELINE_SD * math.sqrt(h - 1)
        else:
            if h == 2:
                smooth = (2.0 * sd_raw[2] + sd_raw[3]) / 3.0
            elif h == 24:
                smooth = (sd_raw[23] + 2.0 * sd_raw[24]) / 3.0
            else:
                smooth = (sd_raw[h - 1] + 2.0 * sd_raw[h] + sd_raw[h + 1]) / 4.0
            sd = min(max(smooth * SIGMA_SCALE, SIGMA_FLOOR), SIGMA_CAP)
            sd_used[h] = sd
            qs = [p1 * math.exp(centres[h] + sd * Z[tau]) for tau in LEVELS]
        rows.append({"hour": h, "q": _post_process(qs)})

    LAST_STATS.clear()
    LAST_STATS.update(
        {
            "p1": p1,
            "usable": usable_count,
            "clean": len(clean),
            "dropped": dropped,
            "sd_raw": dict(sd_raw),
            "sd_used": dict(sd_used),
            "baseline_hours": [h for h in range(2, 25) if is_baseline[h]],
        }
    )
    return {"hours": rows}


def _fallback_case(case):
    """House-baseline forecast for the whole case (used on any error)."""
    p1 = _anchor_p1(case)
    rows = [{"hour": h, "q": _post_process(_baseline_row(p1, h))} for h in range(2, 25)]
    return {"hours": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(input_dir) if f.endswith(".in"))
    for fname in names:
        stem = fname[: -len(".in")]
        out_path = os.path.join(output_dir, stem + ".out")
        try:
            with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            result = forecast_case(case)
        except Exception as exc:  # never crash, never miss an output
            print("error on %s: %s" % (fname, exc), file=sys.stderr)
            try:
                with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as fh:
                    case = json.load(fh)
                result = _fallback_case(case)
            except Exception as exc2:
                print("fallback failed on %s: %s" % (fname, exc2), file=sys.stderr)
                result = {"hours": [{"hour": h, "q": [1000.0] * 5} for h in range(2, 25)]}
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        print("wrote %s" % out_path, file=sys.stderr)


if __name__ == "__main__":
    main()
