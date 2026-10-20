# cycle_factor

**Status:** confirmed — hypothesis rejected. The shared cycle factor is real
in aggregate and unusable at the sector level.

Stage 5 of `docs/plans/forecasting-feature-roadmap.md`.

## Goal

Does a factor shared by every sector in a cycle carry usable signal?
Concretely: does multiplying each sector's own mean by a *forecast* of that
shared factor beat simply using the sector's mean?

## Hypothesis (written before running)

It wins. Total items per cycle swing about ±20% around their mean and the
swing hits every sector at once — most likely the magazine and the
promotions the client sets at each cycle's opening, for which no calendar
exists. Forecasting one short series is far easier than forecasting 633.

## Design note

The candidate is `level × factor`, where the level is each sector's mean
over everything observed — exactly what `naive:mean` forecasts. **A factor
forecast of 1.0 therefore reproduces `naive:mean` exactly**, so every
difference against that row is attributable to the factor and nothing else.

Four factor forecasts are compared: `last` (repeat the most recent observed
factor), `mean` over the last 2 and 3, and `all` (average every factor seen,
which amounts to a constant correction).

## Result (2026-09-22) — the factor makes things worse, every way we forecast it

| candidate | MAE | vs naive:mean |
|---|---|---|
| naive:mean | **1840.0** | — |
| lightgbm ratio (stage 3) | 1859.5 | +19.6 |
| cycle_factor:mean3 | 1926.0 | +86.0 |
| cycle_factor:mean2 | 1933.0 | +93.0 |
| cycle_factor:last1 | 1962.1 | +122.1 |
| cycle_factor:all | 1983.8 | +143.8 |

Every variant loses, and by a wide margin — far wider than anything else in
the roadmap. Two measurements explain it.

### The factor has no memory

The observed factor series, estimated as a ratio of sums against each
sector's running mean:

```
1.48  1.22  1.21  1.20  1.26  0.94  1.12  1.09  1.26  1.18  1.06
```

Its lag-1 autocorrelation is **+0.110** (and +0.031 if the factor is
estimated as a median of ratios instead). **The factor cannot be forecast
from its own past.** Whatever produces the swing — a promotion, a strong
magazine — does not announce itself one cycle ahead, which is consistent
with the client's own description: promotions are decided per cycle
according to business need, and no calendar exists.

So this stage did not fail because the cycle effect is absent. It failed
because the effect is **not predictable**, which is a different and more
final finding.

### And the level of the factor is not a free correction either

Note the series sits mostly above 1. That is a genuine bias: a sector's
running mean systematically under-shoots its next cycle. But correcting it
*raises every forecast*, and MAE is minimised by the median, not the mean.
The lift is driven by large sectors and a right-skewed distribution; the
typical sector does not need it. Hence `cycle_factor:all` — a constant
correction of about 1.17 — is the **worst** variant of the four.

### An estimator bug caught on the way, worth recording

The first implementation estimated the factor as the **mean of
`items / running_mean`** across sectors. That series opened at **3.51** and
decayed toward 1. It looked like a spectacular demand surge; it was an
artifact. Early rows divide by a running mean built from one or two cycles,
and a small noisy denominator blows the ratio up — the mean of ratios is
not the ratio of means.

Switching to a ratio of sums cut the dispersion by a factor of five (σ 0.701
→ 0.138) and improved the candidate from 2014.6 to 1926.0. It still loses.
Both numbers are recorded because the first one is exactly the kind of
result that would have looked like a discovery.

### What this corrects in the roadmap

The roadmap justified this stage with the **total items per cycle** series
(0.62 … 1.20 indexed to its mean). That is a different quantity: it mixes
the cycle effect with changes in which sectors were observed. Measured
properly — relative to what each sector's own history expected — the effect
is smaller and has essentially no memory. The roadmap's framing of stage 5
as promising was based on the looser measurement.

### Decision

Dropped. The candidate stays in the repository as a documented negative
result, because "the shared cycle effect is real but unpredictable" is worth
stating in the paper: it bounds what any aggregate-level correction can do.

## How to run

```bash
uv run python -m experiments.cycle_factor.run
```
