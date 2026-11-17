# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ragilwidiyanto4646/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ragilwidiyanto4646/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (2026-11-19 UTC), hour by hour, from the first hour's viewer count. For each of hours 2..24 it writes five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's concurrent-viewer count, scored by mean pinball loss on the log scale. Entry for the [make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## How it works

The model works in log ratios to the hour-1 print. For each hour 2..24 it fits a per-hour quadratic regression of `log(viewers / hour-1)` on `log(hour-1)` over the case's comparable past big launches, then:

- **Extrapolation clamp** — the fitted curve is evaluated at most at the largest comparable hour-1, so for a launch bigger than every comparable the shape freezes instead of extrapolating (the decisive guard: huge launches flatten rather than fall forever).
- **Kernel-weighted median residual correction** — residuals are corrected by their Gaussian-kernel-weighted median (bandwidth 1.0 in log units), weighting comparables near the case's own hour-1 most.
- **Calibrated spread** — the weighted residual standard deviation, floored at 0.15, feeds the normal quantiles of the five levels.

Hours with fewer than 5 comparable points fall back to the house persistence baseline shape. The program never crashes and always writes a valid output for every input case (baseline fallback on any error). Deterministic, standard library only.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

This reads every `*.in` case in the input dir and writes one `<stem>.out` per case into `out/`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=ztnjpsw2&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
