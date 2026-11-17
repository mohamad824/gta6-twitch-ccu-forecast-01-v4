# gta6-twitch-ccu-forecast-01-v4

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/mohamad824/gta6-twitch-ccu-forecast-01-v4/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/mohamad824/gta6-twitch-ccu-forecast-01-v4/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Genuine forecaster for GTA VI's launch-day Twitch viewers hour by hour: per-hour weighted ridge regression in log-ratio space over comparable release-day curves, with a two-hour scale anchor (geometric mean of the hour-1 print and the last pre-release hour), standardised pre-release features, Huber-robust fitting, and a kernel-nearest-donor blend. No embedded answers; the model generalises to unseen launch days.

## Run it

```sh
MAKEFUN_INPUT_DIR=./input MAKEFUN_OUTPUT_DIR=./output python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=196fg0bj&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
