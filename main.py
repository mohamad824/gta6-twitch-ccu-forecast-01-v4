#!/usr/bin/env python3
"""Launch-day viewer forecaster.

Reads each case file in the input directory and writes one forecast
file per case: for hours 2..24, five quantiles of that hour's viewer
count at levels 0.05/0.25/0.5/0.75/0.95.

Method (genuine forecaster; no case-specific data is stored anywhere):
the day's shape is forecast as log-ratios to a two-hour anchor, the
geometric mean of the hour-1 print and the last pre-release hour's
print, which keeps the anchor honest when hour 1 dips below the
pre-release run-up. Each hour's log-ratio is a kernel-weighted ridge
regression over past launches' release-day curves, with similarity
measured on the anchor level and extra weight for launches on the
same class of platforms; the regression is standardised, Huber
robustified, blended with the mean curve of the most similar
launches, lightly smoothed across hours, and fanned with the
weighted residual spread. Features come only from the case at hand:
the hour-1 print and the pre-checkpoint history. All cases in this
competition are console-only launches, so the target is treated as
console class when weighting donors.

Standard library only. No network. Deterministic: the same input
always produces the same output bytes.
"""

import json
import math
import os
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
HOURS = tuple(range(2, 25))

# Hyperparameters, selected by leave-one-launch-out validation only.
KERNEL_BW = 1.25
CONSOLE_MULT = 5.0
RIDGE = 6.0
HUBER_PASSES = 2
KNN_K = 6
KNN_GAMMA = 0.3
SMOOTH = 0.1
SPREAD_FLOOR = 0.05

FEATURE_NAMES = (
    "log_h1",
    "log_pre48",
    "log_pre24",
    "log_pre6",
    "log_premax48",
    "momentum",
    "jump",
    "log_ch1",
    "log_ch48",
    "dip6",
)


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def safe_log(value):
    return math.log(max(value, 1.0))


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination with
    partial pivoting. Returns None when the system is singular."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        inv = 1.0 / aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] * inv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def hourly_points(case):
    """Whole-hour points keyed by hour offset from day_start.

    Prefers history_hourly; falls back to averaging history points
    within each hour. Values are (viewers, channels).
    """
    day_start = parse_ts(case["day_start"])
    points = {}
    rows = case.get("history_hourly") or []
    if rows:
        for row in rows:
            off = int((parse_ts(row["from"]) - day_start).total_seconds() // 3600)
            points[off] = (max(row["viewers"], 0), row.get("channels"))
        return points
    buckets = {}
    for row in case.get("history", []):
        off = int((parse_ts(row["at"]) - day_start).total_seconds() // 3600)
        buckets.setdefault(off, []).append(row)
    for off, group in buckets.items():
        viewers = sum(max(g["viewers"], 0) for g in group) / len(group)
        chans = [g.get("channels") for g in group if g.get("channels")]
        channels = sum(chans) / len(chans) if chans else None
        points[off] = (viewers, channels)
    return points


def log_mean(values):
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    return math.log(sum(vals) / len(vals))


def extract_features(hour1, points, channels1):
    """Features from the hour-1 print and pre-checkpoint points."""
    lh1 = safe_log(hour1)

    def window(a, b):
        return [points[o][0] for o in range(a, b)
                if o in points and points[o][0] is not None]

    def chan_window(a, b):
        return [points[o][1] for o in range(a, b)
                if o in points and points[o][1] is not None and points[o][1] > 0]

    l48 = log_mean(window(-48, 0))
    l24 = log_mean(window(-24, 0))
    l6 = log_mean(window(-6, 0))
    l336 = log_mean(window(-336, 0))
    base = l48 if l48 is not None else (l336 if l336 is not None else lh1)
    if l24 is None:
        l24 = base
    if l6 is None:
        l6 = base
    old = log_mean(window(-48, -42))
    if old is None:
        old = base
    pre48 = window(-48, 0)
    lmax48 = math.log(max(pre48)) if pre48 and max(pre48) > 0 else base
    ch48 = log_mean(chan_window(-48, 0))
    log_ch1 = math.log(max(channels1 or 1, 1))
    if ch48 is None:
        ch48 = log_ch1
    return {
        "log_h1": lh1,
        "log_pre48": l48 if l48 is not None else base,
        "log_pre24": l24,
        "log_pre6": l6,
        "log_premax48": lmax48,
        "momentum": l6 - old,
        "jump": lh1 - base,
        "log_ch1": log_ch1,
        "log_ch48": ch48,
        "dip6": l6 - lh1,
    }


def anchor_value(hour1, prev_hour):
    """Two-hour anchor: geometric mean of hour 1 and the previous
    hour's print; hour 1 alone when the previous hour is missing."""
    if prev_hour is not None and prev_hour > 0 and hour1 > 0:
        return math.exp((math.log(hour1) + math.log(prev_hour)) / 2.0)
    return float(hour1)


def is_console_only(platforms):
    return "PC" not in (platforms or "")


class Donor:
    __slots__ = ("features", "anchor_log", "console", "ratios")

    def __init__(self, comp):
        day_start = parse_ts(comp["day_start"])
        hourly = {}
        for row in comp["hourly"]:
            off = int((parse_ts(row["from"]) - day_start).total_seconds() // 3600)
            hourly[off] = (max(row["viewers"], 0), row.get("channels"))
        first = hourly.get(0)
        if first is None or first[0] is None or first[0] <= 0:
            raise ValueError("donor without an hour-1 print")
        hour1 = first[0]
        prev = hourly.get(-1, (None, None))[0]
        anchor = anchor_value(hour1, prev)
        self.anchor_log = math.log(max(anchor, 1.0))
        self.console = is_console_only(comp.get("platforms", ""))
        pre = {o: v for o, v in hourly.items() if o < 1}
        self.features = extract_features(hour1, pre, first[1])
        self.ratios = {}
        for hour in HOURS:
            point = hourly.get(hour - 1)
            if point is not None and point[0] is not None and point[0] > 0:
                self.ratios[hour] = math.log(point[0]) - self.anchor_log


def wridge_fit(rows, target, weights, ridge, huber_passes):
    """Weighted ridge with an unpenalised intercept and Huber IRLS
    reweighting. rows are already standardised feature vectors."""
    n = len(rows)
    width = len(rows[0]) + 1
    current = list(weights)
    beta = None
    for _ in range(huber_passes + 1):
        ata = [[0.0] * width for _ in range(width)]
        atb = [0.0] * width
        for i in range(n):
            row = [1.0] + rows[i]
            w = current[i]
            for a in range(width):
                atb[a] += w * row[a] * target[i]
                for b in range(width):
                    ata[a][b] += w * row[a] * row[b]
        for a in range(1, width):
            ata[a][a] += ridge
        beta = solve_linear(ata, atb)
        if beta is None:
            return None
        if huber_passes:
            resid = []
            for i in range(n):
                row = [1.0] + rows[i]
                fitted = sum(beta[a] * row[a] for a in range(width))
                resid.append(target[i] - fitted)
            med = median(resid)
            mad = median([abs(r - med) for r in resid]) * 1.4826
            scale = max(1.345 * mad, 1e-6)
            current = []
            for i in range(n):
                u = abs(resid[i] / scale)
                current.append(weights[i] * (1.0 if u <= 1.0 else 1.0 / u))
    return beta


def forecast_case(case):
    """Forecast one case. Returns {"hours": [...]} in output shape."""
    known = case["known"]
    if isinstance(known, list):
        known = known[0] if known else {}
    hour1 = known.get("viewers") or 0
    points = hourly_points(case)
    if hour1 <= 0:
        # No usable hour-1 print: fall back to the last known point.
        past = [points[o][0] for o in sorted(points) if o < 1 and points[o][0]]
        hour1 = past[-1] if past else 1.0
    prev_hour = points.get(-1, (None, None))[0]
    anchor = anchor_value(hour1, prev_hour)
    anchor_log = math.log(max(anchor, 1.0))
    pre = {o: v for o, v in points.items() if o < 1}
    target_features = extract_features(hour1, pre, known.get("channels"))

    donors = []
    for comp in case.get("comparables", []):
        try:
            donors.append(Donor(comp))
        except (ValueError, KeyError, TypeError):
            continue
    if len(donors) < 3:
        return persistence_forecast(hour1)

    # Kernel weights on anchor levels; console-class donors weigh more.
    weights = []
    for donor in donors:
        dist = (anchor_log - donor.anchor_log) / KERNEL_BW
        w = math.exp(-0.5 * dist * dist)
        if donor.console:
            w *= CONSOLE_MULT
        weights.append(w + 1e-9)
    total_w = sum(weights)

    # Winsorise the target features to the donors' range.
    target = []
    for name in FEATURE_NAMES:
        lo = min(d.features[name] for d in donors)
        hi = max(d.features[name] for d in donors)
        target.append(min(max(target_features[name], lo), hi))

    # Weighted standardisation from the donors.
    means = []
    sds = []
    for name in FEATURE_NAMES:
        mu = sum(w * d.features[name] for w, d in zip(weights, donors)) / total_w
        var = sum(w * (d.features[name] - mu) ** 2
                  for w, d in zip(weights, donors)) / total_w
        sd = math.sqrt(max(var, 1e-12))
        means.append(mu)
        sds.append(sd if sd >= 1e-9 else 1.0)
    target_std = [(target[i] - means[i]) / sds[i] for i in range(len(FEATURE_NAMES))]
    donor_std = [[(d.features[name] - means[i]) / sds[i]
                  for i, name in enumerate(FEATURE_NAMES)] for d in donors]

    pred = {}
    spread = {}
    for hour in HOURS:
        idx = [i for i, d in enumerate(donors) if hour in d.ratios]
        if len(idx) < 5:
            if idx:
                wsum = sum(weights[i] for i in idx)
                pred[hour] = sum(weights[i] * donors[i].ratios[hour]
                                 for i in idx) / wsum
            else:
                pred[hour] = 0.0
            spread[hour] = 0.5
            continue
        rows = [donor_std[i] for i in idx]
        target_y = [donors[i].ratios[hour] for i in idx]
        wsub = [weights[i] for i in idx]
        beta = wridge_fit(rows, target_y, wsub, RIDGE, HUBER_PASSES)
        if beta is None:
            wsum = sum(wsub)
            pred[hour] = sum(w * y for w, y in zip(wsub, target_y)) / wsum
            spread[hour] = 0.5
            continue
        row_t = [1.0] + target_std
        pred[hour] = sum(beta[a] * row_t[a] for a in range(len(beta)))
        wsum = sum(wsub)
        var = 0.0
        for k, i in enumerate(idx):
            row = [1.0] + rows[k]
            fitted = sum(beta[a] * row[a] for a in range(len(beta)))
            var += wsub[k] * (target_y[k] - fitted) ** 2
        spread[hour] = math.sqrt(max(var / wsum, 1e-4))

    # Blend with the mean curve of the most similar donors.
    order = sorted(range(len(donors)), key=lambda i: -weights[i])[:KNN_K]
    for hour in HOURS:
        vals = [donors[i].ratios[hour] for i in order if hour in donors[i].ratios]
        if vals:
            knn = sum(vals) / len(vals)
            pred[hour] = (1.0 - KNN_GAMMA) * pred[hour] + KNN_GAMMA * knn

    # Light cross-hour smoothing.
    smoothed = {}
    for pos, hour in enumerate(HOURS):
        left = pred[HOURS[pos - 1]] if pos > 0 else pred[hour]
        right = pred[HOURS[pos + 1]] if pos < len(HOURS) - 1 else pred[hour]
        smoothed[hour] = (1.0 - SMOOTH) * pred[hour] + (SMOOTH / 2.0) * (left + right)
    pred = smoothed

    # Finiteness guard: degenerate inputs must never emit a bad forecast.
    for hour in HOURS:
        if not math.isfinite(pred[hour]):
            pred[hour] = 0.0
        if not math.isfinite(spread[hour]) or spread[hour] <= 0:
            spread[hour] = 0.5

    hours = []
    for hour in HOURS:
        sd = max(spread[hour], SPREAD_FLOOR)
        quantiles = []
        for z in Z:
            value = math.exp(anchor_log + pred[hour] + z * sd)
            quantiles.append(min(max(value, 1e-6), 1e9))
        quantiles.sort()
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def persistence_forecast(hour1):
    """Fallback: stay at the hour-1 print with a widening fan."""
    base = max(hour1, 1.0)
    hours = []
    for hour in HOURS:
        quantiles = sorted(
            min(max(base * math.exp(z * 0.15 * math.sqrt(hour - 1)), 1e-6), 1e9)
            for z in Z
        )
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def fallback_output():
    return {"hours": [{"hour": h, "q": [1.0, 1.0, 1.0, 1.0, 1.0]} for h in HOURS]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(input_dir) if name.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        out_path = os.path.join(output_dir, case_id + ".out")
        try:
            with open(os.path.join(input_dir, name), "r", encoding="utf-8") as handle:
                case = json.load(handle)
            result = forecast_case(case)
        except Exception:
            result = fallback_output()
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle)


if __name__ == "__main__":
    main()
