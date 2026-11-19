# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/artezzy969-eth/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/artezzy969-eth/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one — make.fun bounty entry. A Python 3.13 standard-library-only program that reads a case (`makefun/gta6-trophies-case/v1`) and forecasts every trophy's Exophase earn rate at the target as five quantiles, scored by mean pinball loss on the logit scale.

## Model

Exophase refreshes its counts on its own schedule (hours apart), so rates are step functions: nothing moves between refreshes, then both the tracked-player count `P` and each trophy's earner count `E` jump. Since a rate is `E / P`, the program models the two sides separately and fits both at runtime from the case's own history:

- `log P` and `log E_i` are each fit by least squares against `ln(1 + hours since the first read)` and extrapolated to the target; the median forecast is `100 * E_target / P_target`.
- Per-trophy earner slopes are shrunk toward the pooled (median) slope across trophies, because early earner counts are small integers and single-trophy fits are noisy.
- When a history is flat (no refresh seen yet, as on launch night), slopes fall back to priors from the refresh aggregates published in the competition spec (638 → 3,868 players tracked, 943 → 4,276 trophies earned over the first ~4.7 h). The starter trophy (100% from the first read) stays at 100%.
- The logit spread combines a random-walk term, the fit residual, and a refresh-jump term that grows with the expected log-P growth over the horizon — a refresh landing before the target moves every rate at once, so the spread is honest about that.

Local practice validation (against an aggregate-consistent proxy answer derived from the spec's published refresh totals, not the server's true answers): loss ≈ 0.10 vs the house persistence baseline's ≈ 1.40, typical miss ≈ 0.01 pp, 90% coverage ≈ 0.98.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `*.out` per case. Deterministic (two runs are byte-identical), no network, well under the 8 s per-case cap.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
