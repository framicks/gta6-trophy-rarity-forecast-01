# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/framicks/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/framicks/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A refresh-aware forecaster for Grand Theft Auto VI (PS5) trophy earn rates on Exophase at 2026-11-26T00:00:00Z, entered in the "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## Approach

Exophase refreshes a game's counts on its own schedule, not with every player's trophy: between refreshes the printed rates do not move at all, and at a refresh the tracked-player count jumps while most rates fall, because the new players have mostly not earned each trophy yet. Persistence is therefore wrong exactly when a refresh lands — the house baselines score about 1.51 on the practice cases for this reason. This forecaster models the refresh instead of ignoring it:

- **Trend branch (the final case):** when the history shows real refreshes, it collapses runs of identical reads into distinct refresh points and, per trophy, fits logit(rate) against ln(1 + hours since release), weighted toward recent reads. The slope used for extrapolation is the *recent* (last-36h) slope shrunk toward its tier's median, damped with distance and capped, because a trophy's rate saturates as players finish it. The forecast is anchored at the last observed rate and aimed at the target less half a refresh gap, since the answer is the page as of Exophase's last refresh before the target. The spread comes from the fit's residuals plus an extrapolation penalty that grows with horizon.
- **Flat-history branch (the practice cases):** when every read is identical there is no trend to fit, so it applies a first-refresh prior: the median rate is the last rate times a dilution factor (about 0.18–0.40 by trophy kind — story, grind, collectible, easy, or mid — from the launch-night refresh, where players tracked went 638 → 3,868 while every rate but the 100% one fell), with a wide spread covering both "a refresh lands" and "it does not". A trophy at 100% stays at ~100%.
- **Honest quantiles:** the 5%–95% band is calibrated from backtests (see below), widening with horizon — the final's 4.5-day horizon gets a much wider spread than the practice's 90 minutes.

Backtests (own evaluation, scoring exactly as RULES.txt defines it): on simulated first-refresh answers for the practice cases the forecaster scores 0.25–0.32 mean pinball loss against the starter's 0.88–1.69 across five refresh scenarios; on a synthetic multi-day final (2.5 days of history, 4.5 days ahead) it scores 0.24 against the starter's 0.30, and 0.22 against 0.42 in a slow-saturation variant.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Put the case files (`<id>.in`, from the competition kit's `input/`) in `input/` first. The grader instead runs `python3 -P -s /bot/main.py` with `/input` and `/output` unset in the environment.

Python 3.13, standard library only, no network. Reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: one row per trophy with five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of its earn rate at the target, rounded to 4 decimals half-up. Deterministic — the same input always produces byte-identical output — and runs in well under a second per case.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
