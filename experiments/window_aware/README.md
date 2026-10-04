# window_aware

**Status:** confirmed.

> **Since 2026-10-04 the window is always on**, for these two models and
> every other candidate. The window-blind rows ("ETS", "LightGBM") were
> dropped from the config, so a rerun reproduces only the window-aware rows
> and the β = 1 row. Their numbers stay in the tables below.
> `plot_level.py` now asks about each sector's window from the same cycle
> one year earlier, shifted 52 weeks.

**Goal:** the harness now answers f(sector, opening day, cycle length) (see
`src/forecasting/README.md`). This experiment asks whether the ETS and
LightGBM forecast better once they are told the scenario window: its length,
the days since the sector's previous opening, and the day of the month.

```bash
uv run python -m experiments.window_aware.run
```

The run takes about 3 minutes on `base_tratada_v2`. It uses the same split as
`v2_baseline`: 11 folds, one cycle ahead.

## What each model does with the window

- **ETS (`cycle_days_exponent`)** assumes a cycle's items scale with its length
  as `items ∝ cycle_days ** β`.
  - It smooths the level per unit of that scale, then multiplies by the queried
    length to the power β.
  - β = 0 ignores the length. β = 1 is "items per day × days".
  - `pooled` picks β together with α, from the training history, over a grid
    from 0 to 1 in steps of 0.1.
- **LightGBM (`window_features: true`)** gets three extra features:
  - `cycle_days`: the window's length;
  - `days_since_previous_opening`: how many days since this sector's last
    window opened;
  - `opening_day_of_month`: the day of the month the window opens.

  How far the window starts after the cycle's own opening is **left out on
  purpose**. That offset is the sub-block itself, so a model keyed on it would
  learn "sectors that sat in late sub-blocks" instead of what moving a sector
  does.

Before running, a look at the data set the expectation. Relative to each
sector's own mean, 14-day windows sell 0.97 and 21-day windows sell 1.01. A
per-day rate would predict 14 / 21 = 0.67 for the short windows.

## Hypotheses (written before running)

1. The pooled β lands at 0.2 or below. The ETS with it lands within 0.5% of the
   plain pooled ETS, per cycle and per CD-day.
2. Forcing β = 1 is more than 5% worse per cycle than the plain pooled ETS.
3. LightGBM with the window features lands within 2% of LightGBM without them,
   per cycle.

## Result (2026-09-28)

The table covers 6,736 points. The two reference rows reproduce
`v2_baseline` exactly.

| candidate | MAE per CD-day | MAE per sector-day | MAE per cycle | vs. same model without the window |
|---|---|---|---|---|
| **LightGBM + janela** | **5,384** | **264.6** | **1,802** | −0.4% CD-day, −0.7% cycle |
| LightGBM | 5,403 | 267.0 | 1,814 | — |
| ETS, β pooled | 5,413 | 284.6 | 1,829 | −0.2% CD-day, −0.3% cycle |
| ETS | 5,423 | 283.9 | 1,834 | — |
| ETS, β = 1 | 5,653 | 291.1 | 1,886 | +4.2% CD-day, +2.8% cycle |

The pooled β chosen at each fold was 0.1 in the first six and 0.2 in the last
five. α stayed between 0.15 and 0.19.

- **Hypothesis 1 is upheld.** Cycle length matters, but only a little. With
  β = 0.2, a 14-day window is forecast (14 / 21)^0.2 = 0.92 of a 21-day one,
  not 0.67.
- **Hypothesis 2 is refuted in size, not in direction.** β = 1 is worse, but
  by 2.8% per cycle, not by more than 5%. Per CD-day it is 4.2% worse.
- **Hypothesis 3 is upheld.** The window features help LightGBM by 0.7% per
  cycle and 0.4% per CD-day.

### Where the gain comes from: the short windows

The overall gains are small because short windows are rare: 1,156 of the
6,736 points have 16 days or fewer. Split by window length:

| candidate | MAE, short (≤ 16 d) | bias, short | MAE, normal | bias, normal |
|---|---|---|---|---|
| ETS | 2,100 | +366 | 1,769 | −66 |
| ETS, β pooled | 2,055 | +215 | 1,773 | −1 |
| ETS, β = 1 | **1,913** | −745 | 1,871 | +499 |
| LightGBM | 2,043 | +143 | **1,759** | −777 |
| LightGBM + janela | 1,971 | −522 | 1,760 | −749 |

- **On short windows, knowing the length cuts the error.** LightGBM improves by
  3.5% and the ETS with pooled β by 2.1%. On normal windows neither model
  changes.
- **β = 1 is best on short windows and worst on normal ones.** Its level is
  kept per day, so every 14-day cycle in the history, divided by 14, pushes
  the level up. That inflates every 21-day forecast (+499 bias). A single β
  cannot fit both at once, and the pooled β picks the compromise.
- **With the window, LightGBM's bias on short windows flips from +143 to
  −522.** It now separates short from normal windows, but overshoots the
  cut.

## Scenario check (not scored)

Using all the history, I asked both models about cycle 202613 for the three
largest sectors of the last cycle: five opening days (±2 around a regular
21-day spacing) × two lengths (14 and 21 days).

- **The opening day changed nothing.** Across the five days, both models gave
  the same value for each sector. The gap since the last opening and the day
  of the month vary too little over ±2 days to cross any of LightGBM's split
  points.
- **Only the length changes the answer.** For sector 3314, a 14-day window
  instead of a 21-day one gives −8% on both models: the ETS forecasts 7,517
  against 8,152, and LightGBM 6,613 against 7,185.

For the optimizer, this means the difference between opening days comes almost
entirely from the shape (how the total spreads over the days), not from the
total.

## Caveats

- **The backtest only scores the window that actually happened.** A win here
  shows the window features are not noise. It does not show that the answers
  for opening days that never happened are right.
- **The length is almost cycle-wide.** Most 14-day windows belong to cycles
  that are short for everyone (202605, 202607, …). So "14 days" partly stands
  in for "that particular cycle".
- **One LightGBM seed.** A 0.4% CD-day difference is within the seed noise
  measured in earlier experiments. The short-window split (−3.5%) is the more
  solid number.
