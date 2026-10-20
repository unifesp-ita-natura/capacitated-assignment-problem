# Makridakis, Spiliotis & Assimakopoulos — M5 accuracy competition: Results, findings and conclusions

**Link/DOI:** https://www.sciencedirect.com/science/article/pii/S0169207021001874
**Read by:** Claude (via WebSearch summary, not full text — flag for a human
deep-read before relying on this for a final method choice)
**Date:** 2026-09-08

## Summary

Retrospective on the M5 forecasting competition (retail sales, tens of
thousands of series). All top-performing methods were "pure" ML approaches
(mostly LightGBM-based gradient boosting, some N-BEATS/seq2seq neural nets),
significantly beating classical statistical benchmarks (ARIMA/ETS) and their
combinations — the first M-competition where that happened. A key driver was
**cross-learning**: training one shared model across many series beat fitting
an independent local model per series, especially for short/noisy bottom-level
series.

## Relevant to us because

Directly backs two decisions in `docs/papers/forecasting-volume-spec.tex`:
recommending a single global LightGBM model with `sector_id`/region as
categorical features (Section 2.3) instead of ~800 independent per-sector
models, and keeping ARIMA/ETS/Prophet as baselines to beat rather than
assuming they're the strongest candidates by default (Section 2.1).

## Open questions / follow-ups

- M5's series (thousands, retail SKUs) are a different scale/domain than our
  ~800 sector-cycle series — the cross-learning advantage should be verified
  empirically here, not assumed to transfer directly.
- A human should read the full paper (this note is based on a search-engine
  summary) before citing it in the Overleaf submission.
