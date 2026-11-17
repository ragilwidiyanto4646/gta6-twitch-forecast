"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Reads one make.fun case file per run (from $MAKEFUN_INPUT_DIR or /input/),
forecasts hours 2..24 as five quantiles each, and writes
$output/<stem>.out for every *.in file found.

Model: per-hour quadratic regression of log(viewers/hour-1) on
log(hour-1) over the case's comparable launches, evaluated with an
extrapolation clamp (never beyond the largest comparable hour-1), a
kernel-weighted median residual correction, and a weighted residual
standard deviation floored at 0.15. Quantiles come from the normal
z-scores of the five levels. Python 3.13, standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

Z_SCORES = (
    -1.6448536269514722,
    -0.6744897501960817,
    0.0,
    0.6744897501960817,
    1.6448536269514722,
)

MIN_SD = 0.15
MIN_POINTS = 5
KERNEL_BANDWIDTH = 1.0


def _parse_time(text):
    """Parse an ISO-8601 timestamp (with a trailing Z) into an aware datetime."""
    if not isinstance(text, str):
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _round_half_up(value):
    """Round a float to 2 decimals, half up, via its decimal string."""
    return float(Decimal(repr(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _post_process(quantiles):
    """Enforce non-decreasing, clamp to [0.01, 1e9], round half-up to 2 dp."""
    out = []
    prev = None
    for q in quantiles:
        if not (math.isfinite(q)):
            q = 0.01
        if prev is not None and q < prev:
            q = prev
        q = min(max(q, 0.01), 1e9)
        q = _round_half_up(q)
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def _baseline_row(p1, hour):
    """House persistence baseline shape for one hour."""
    spread = 0.15 * math.sqrt(hour - 1)
    return _post_process([p1 * math.exp(z * spread) for z in Z_SCORES])


def _anchor_p1(case):
    """Hour-1 print, with fallbacks to the last positive history point."""
    known = case.get("known") or []
    p1 = None
    if known:
        row = None
        for candidate in known:
            if isinstance(candidate, dict) and candidate.get("hour") == 1:
                row = candidate
                break
        if row is None:
            row = known[0]
        if isinstance(row, dict):
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                p1 = float(viewers)
    if p1 is None:
        for key in ("history_hourly", "history"):
            points = case.get(key) or []
            for point in reversed(points):
                if isinstance(point, dict):
                    viewers = point.get("viewers")
                    if isinstance(viewers, (int, float)) and viewers > 0:
                        p1 = float(viewers)
                        break
            if p1 is not None:
                break
    if p1 is None:
        p1 = 1000.0
    return p1


def _comparable_curves(case):
    """Usable comparables: name -> {hour: viewers} for hours 1..24.

    A comparable is usable only if it has an hour-1 point with
    viewers > 0; that automatically drops the GTA VI pre-release
    history comparable. Sorted by name for determinism.
    """
    usable = []
    for comp in case.get("comparables") or []:
        if not isinstance(comp, dict):
            continue
        start = _parse_time(comp.get("day_start"))
        if start is None:
            continue
        curve = {}
        for point in comp.get("hourly") or []:
            if not isinstance(point, dict):
                continue
            viewers = point.get("viewers")
            if viewers is None or not isinstance(viewers, (int, float)):
                continue
            if viewers <= 0:
                continue
            at = _parse_time(point.get("from"))
            if at is None:
                continue
            hour = int((at - start).total_seconds() // 3600) + 1
            if 1 <= hour <= 24 and hour not in curve:
                curve[hour] = float(viewers)
        if 1 in curve:
            usable.append((comp.get("name") or "", curve))
    usable.sort(key=lambda item: item[0])
    return usable


def _fit_quadratic(points):
    """Fit y ~ a + b1*x + b2*(x-mx)^2 (centered) on (x, y) points.

    Returns a callable f(x). Falls back to a linear fit when the
    centered normal equations are degenerate.
    """
    n = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx = sum(xs) / n
    my = sum(ys) / n
    m2 = sum((x - mx) ** 2 for x in xs) / n
    us = [x - mx for x in xs]
    vs = [(x - mx) ** 2 - m2 for x in xs]
    ws = [y - my for y in ys]
    s11 = sum(u * u for u in us)
    s22 = sum(v * v for v in vs)
    s12 = sum(u * v for u, v in zip(us, vs))
    t1 = sum(u * w for u, w in zip(us, ws))
    t2 = sum(v * w for v, w in zip(vs, ws))
    det = s11 * s22 - s12 * s12
    if abs(det) > 1e-12:
        b1 = (t1 * s22 - t2 * s12) / det
        b2 = (t2 * s11 - t1 * s12) / det
        a = my - b1 * mx - b2 * m2

        def fitted(x, a=a, b1=b1, b2=b2, mx=mx):
            return a + b1 * x + b2 * (x - mx) ** 2

        return fitted
    if s11 > 0:
        b = sum(u * w for u, w in zip(us, ws)) / s11
    else:
        b = 0.0
    a = my - b * mx

    def fitted_linear(x, a=a, b=b):
        return a + b * x

    return fitted_linear


def _weighted_median(pairs):
    """Weighted median of (value, weight) pairs, sorted by value."""
    ordered = sorted(pairs, key=lambda item: item[0])
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[0][0] if ordered else 0.0
    half = total / 2.0
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= half:
            return value
    return ordered[-1][0]


def _forecast_hour(p1, hour, points):
    """One hour's five quantiles, given (x=log hour-1, y=log ratio) points."""
    log_p1 = math.log(p1)
    if len(points) < MIN_POINTS:
        return _baseline_row(p1, hour)
    fitted = _fit_quadratic(points)
    residuals = [y - fitted(x) for x, y in points]
    x_max = max(x for x, _ in points)
    x_eval = min(log_p1, x_max)
    weights = [
        math.exp(-0.5 * ((x - log_p1) / KERNEL_BANDWIDTH) ** 2)
        for x, _ in points
    ]
    correction = _weighted_median(list(zip(residuals, weights)))
    center = fitted(x_eval) + correction
    total_w = sum(weights)
    if total_w > 0:
        var = sum(
            w * (r - correction) ** 2 for r, w in zip(residuals, weights)
        ) / total_w
        sd = math.sqrt(var) if var > 0 else 0.0
    else:
        sd = 0.0
    sd = max(sd, MIN_SD)
    quantiles = [p1 * math.exp(center + sd * z) for z in Z_SCORES]
    return _post_process(quantiles)


def forecast_case(case):
    """Forecast one case dict -> {"hours": [{hour, q}, ...]} for hours 2..24."""
    p1 = _anchor_p1(case)
    comparables = _comparable_curves(case)
    # Per hour, the (log hour-1, log ratio) points over comparables
    # that have both an hour-1 and an hour-h point.
    rows = []
    for hour in range(2, 25):
        points = []
        for _name, curve in comparables:
            if hour in curve:
                points.append((math.log(curve[1]), math.log(curve[hour] / curve[1])))
        rows.append({"hour": hour, "q": _forecast_hour(p1, hour, points)})
    return {"hours": rows}


def _validate_forecast(forecast):
    """Raise ValueError unless the forecast has the exact required shape."""
    if not isinstance(forecast, dict) or list(forecast.keys()) != ["hours"]:
        raise ValueError("forecast must have exactly the key 'hours'")
    rows = forecast["hours"]
    if not isinstance(rows, list) or len(rows) != 23:
        raise ValueError("forecast must have exactly 23 rows")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or list(row.keys()) != ["hour", "q"]:
            raise ValueError("row %d has the wrong keys" % index)
        if row["hour"] != index + 2:
            raise ValueError("row %d has hour %r" % (index, row["hour"]))
        q = row["q"]
        if not isinstance(q, list) or len(q) != 5:
            raise ValueError("row %d must have 5 quantiles" % index)
        for value in q:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("row %d has a non-finite quantile" % index)
            if not 0 < value <= 1e9:
                raise ValueError("row %d has an out-of-range quantile" % index)
        if any(q[i] > q[i + 1] for i in range(4)):
            raise ValueError("row %d quantiles are not non-decreasing" % index)


def _fallback_forecast(case):
    """Baseline-only forecast used when anything in a case fails."""
    try:
        p1 = _anchor_p1(case)
    except Exception:
        p1 = 1000.0
    return {
        "hours": [
            {"hour": hour, "q": _baseline_row(p1, hour)}
            for hour in range(2, 25)
        ]
    }


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(output_dir, exist_ok=True)
    try:
        names = sorted(os.listdir(input_dir))
    except OSError as exc:
        print("cannot list input dir: %s" % exc, file=sys.stderr)
        return 0
    for name in names:
        if not name.endswith(".in"):
            continue
        stem = name[:-3]
        in_path = os.path.join(input_dir, name)
        out_path = os.path.join(output_dir, stem + ".out")
        case = {}
        try:
            with open(in_path, "r", encoding="utf-8") as handle:
                case = json.load(handle)
        except Exception as exc:
            print("case %s unreadable (%s); using baseline fallback" % (name, exc),
                  file=sys.stderr)
        try:
            forecast = forecast_case(case)
            _validate_forecast(forecast)
        except Exception as exc:
            print("case %s failed (%s); using baseline fallback" % (name, exc),
                  file=sys.stderr)
            forecast = _fallback_forecast(case)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(forecast, handle, separators=(",", ":"))
        print("wrote %s" % out_path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
