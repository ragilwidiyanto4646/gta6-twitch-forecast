# GTA VI Twitch Forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ragilwidiyanto4646/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ragilwidiyanto4646/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch viewership hour by hour (hours 2–24) as five quantiles each, starting from the hour-1 print. It learns the shape of a launch day from comparable AAA game launches and scores by mean pinball loss on the log scale — the goal is to beat the house persistence baseline and land as close to the real launch curve as possible.

## How it works

Everything runs in log ratios to the hour-1 print. First, each comparable launch is filtered to a clean set: early-access-contaminated, weak-analogue and target-history tiers are dropped, as is any launch whose hour-1 print is under 3% of its own release-day peak (the simultaneous-global-unlock shape, a different regime from GTA VI's rolling-midnight launch). Then, per hour, a log-ratio ridge regression (penalty λ = 0.1 on the slope only) fits how that hour's viewership scales with launch size across the clean comparables. The fitted line is evaluated at most at the largest clean hour-1 — beyond the data the shape freezes instead of extrapolating (extrapolation clamp) — and a kernel-weighted median residual correction (Gaussian kernel, bandwidth 1.0 in log units) recentres it on launches near the target's size. Uncertainty is a per-hour local sigma: the kernel-weighted residual spread, smoothed across neighbouring hours with a centred (1,2,1)/4 moving average, then scaled by 0.8 and guarded by a floor of 0.08 and a cap of 0.9, because residual spread is not homogeneous — small launches are noisy, big launches are tight, and midday hours have the widest gaps. Hours with too few comparables fall back to the house baseline shape. The program is deterministic, stdlib-only, and never crashes: any case that fails is retried with the baseline forecast so a valid output is always written.

## Run it

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` case in the input directory and writes one `<stem>.out` forecast per case into the output directory.

<sub>Entered on [make.fun](https://make.fun/?ref=ztnjpsw2&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
