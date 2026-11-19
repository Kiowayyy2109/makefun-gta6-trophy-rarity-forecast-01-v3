# makefun-gta6-trophy-rarity-forecast-01-v3

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kiowayyy2109/makefun-gta6-trophy-rarity-forecast-01-v3/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kiowayyy2109/makefun-gta6-trophy-rarity-forecast-01-v3/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An adaptive-spread calibration forecaster for GTA VI trophy earn rates on Exophase. It compresses each case's read history into distinct page states (the page only changes when Exophase refreshes it), fits saturating log-time growth curves for the tracked player base and for each trophy's earners with hierarchical trophy-to-tier-to-global slope shrinkage, and blends that structural earners/players view with persistence according to how much history and horizon the case has. Its logit-space spread is calibrated from the case's own observed refresh-to-refresh volatility rather than a fixed constant, so a flat, stable history earns a narrow persistence-like band while a volatile one widens. When the history shows no refresh yet, the pending first refresh is priced as an explicit one-sided dilution mixture: the tracked base is expected to multiply several-fold while earners grow only with playtime, shifting the forecast down asymmetrically instead of widening symmetrically around the last rate. Pure Python 3.13 standard library, deterministic, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---
<sub>Entered on [make.fun](https://make.fun/?ref=8b9bbsfs&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
