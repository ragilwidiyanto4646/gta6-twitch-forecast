# GTA VI Twitch Forecast

[![make.fun bounty](https://img.shields.io/badge/make.fun-bounty-blue)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![stdlib only](https://img.shields.io/badge/deps-stdlib--only-green)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), hour by hour, from the first hour's
viewer count. For each of hours 2..24 it writes five quantiles (5%, 25%,
50%, 75%, 95%) of that hour's concurrent-viewer count, scored by mean
pinball loss on the log scale. Entry for the
[make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## How it works

The model works in log ratios to the hour-1 print. For each hour 2..24 it
fits a quadratic regression of `log(viewers / hour-1)` on `log(hour-1)` over
the case's comparable past big launches, then:

- **Extrapolation clamp** — the fitted curve is evaluated at most at the
  largest comparable hour-1, so for a launch bigger than every comparable
  the shape freezes instead of extrapolating (the decisive guard: huge
  launches flatten rather than fall forever).
- **Kernel-weighted median residual correction** — residuals are corrected
  by their Gaussian-kernel-weighted median (bandwidth 1.0 in log units),
  weighting comparables near the case's own hour-1 most.
- **Calibrated spread** — the weighted residual standard deviation,
  floored at 0.15, feeds the normal z-scores of the five levels.

Hours with fewer than 5 comparable points fall back to the house persistence
baseline shape. The program never crashes and always writes a valid output
for every input case (baseline fallback on any error). Deterministic,
standard library only.

## Run locally

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

This reads every `*.in` case in the input dir and writes one
`<stem>.out` per case into `out/`.
