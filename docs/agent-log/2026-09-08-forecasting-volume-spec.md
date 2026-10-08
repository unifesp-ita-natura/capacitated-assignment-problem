# Forecasting volume spec ($L_{i,c}$): literature study and Overleaf-bound write-up

**Date:** 2026-09-08
**Related:** uncommitted — see `docs/papers/forecasting-volume-spec.tex`

## Task

Study forecasting techniques used in the literature for per-sector,
per-cycle volume prediction (ARIMA, SARIMA, ETS, Prophet, XGBoost/gradient
boosting, and others), and produce a document — meant to go into the team's
Overleaf paper — covering: (1) a formal mathematical definition of
$L_{i,c}$ and its boundary with the cycle "shape" $s_{i,t}$; (2) the model
candidates to test with their specific parameters/orders and the level
(sector/block/region) at which each runs; (3) the comparison criterion
between candidates (error metric, validation scheme, decision rule).

## Outcome

A self-contained LaTeX document
(`docs/papers/forecasting-volume-spec.tex`) now specifies all three required
points, grounded in this repo's existing code (`block_assignment.py`'s
notation, `ForecastConfig`/`ForecastResult` scaffolding already added by
`luizamfsantos`, and the synthetic generator's `orders` column) plus a short
literature scan. This repo has no Overleaf integration, so the `.tex` file
is the deliverable's repo-side source — a human still needs to copy/`\input`
it into the actual Overleaf project.

## What changed

- `docs/papers/forecasting-volume-spec.tex` (new) — the spec itself.
- `docs/papers/README.md` (new) — explains the `docs/papers/` convention
  (mirrors `docs/literature/`'s repo-note pattern) since `formal/README.md`
  already referenced this directory but it didn't exist yet.
- `README.md` — added a bullet for `docs/papers/` under "What's here".
- `docs/literature/hyndman-foresight-forecast-accuracy-metrics.md`,
  `docs/literature/hyndman-athanasopoulos-fpp3.md`,
  `docs/literature/makridakis-m5-accuracy-competition.md` (new) — notes for
  the three sources actually cited in the spec.
- `docs/literature/README.md` — index updated with the three notes above.

## Notes

- Both the Semantic Scholar and Zotero MCP servers failed to connect this
  session (`uv` not on `$PATH` for the sandboxed environment running the
  agent) — literature search used WebSearch instead, per
  `docs/literature/README.md`'s fallback guidance. The three notes added are
  based on search-engine summaries, not a full read of the source, and are
  **not yet added to the shared Zotero library** — a human with a working
  Zotero MCP connection (or the desktop client) should add them and file
  under `Forecasting > Time Series Forecasting (Volume Prediction)`.
- Key modeling decision documented but explicitly left open pending real
  data: whether $L_{i,c}$'s unit should be orders (what the synthetic
  generator currently emits), volume/boxes (Luiza's instinct in the original
  task), or items — the spec argues it must match whatever unit
  `daily_capacity` is measured in, in `block_assignment.py`, but that
  requires confirming the real demand base schema.
- Flagged a symbol collision: `block_assignment.py` uses lowercase `c` for
  distribution-center index, while $L_{i,c}$'s own established notation uses
  `c` for cycle — both are internally consistent in their own document but
  will collide if merged verbatim into one Overleaf document without
  renaming one of them.
- Recommended treating LightGBM (already scaffolded in
  `src/config/schema.py`'s `ForecastParams`) as the production
  gradient-boosting implementation and XGBoost only as a one-off sensitivity
  check, rather than maintaining both in parallel.
- No code was changed — `src/forecasting/model.py` and `features.py` remain
  empty stubs; this was a specification/literature task, not an
  implementation task.

## Addendum (2026-09-11)

The author shared the actual paper (`desafio_natura.tex`, Luiz Salles-Neto)
and the demand base's columns, which settled three things the original spec
had left open, and `forecasting-volume-spec.tex` was rewritten accordingly —
it is now a set of labelled blocks to paste into the main paper, not a
standalone spec.

- **Unit resolved: volume.** The base's only quantity column is
  `total_volumes_mascarado` — no orders column, no items column. So Luiza's
  instinct was right and the paper's §3.2 (which reads "quantidade de
  **pedidos**") is wrong and needs correcting. Values are masked, so
  `Cap_{c,a}` must be on the same scale for constraint (3.5.4) to mean
  anything.
- **Notation aligned to the paper.** The paper uses `s` (sector), `d`
  (slot), `a` (day) and `c` (**CD**), so `L_{i,c}` was wrong: it is now
  `L_{s,k}` with `k` for cycle, and the shape is `σ_{s,t}`, not `s_{i,t}`.
- **Granularity gap found.** The base is per cycle (possibly per week via
  `semana_pedido`), but the MIP needs daily `q_{s,a,d}` because `Cap_{c,a}`
  is daily and `z_max`/`z_min` are daily extremes. The daily shape cannot
  come from this base. Raised as a team-level decision (finer source vs.
  weekly reformulation vs. assumed intra-week profile), not something the
  volume-forecasting side can resolve alone.
- **Croston/SBA dropped** from the candidate list: it was there for the
  order-count-with-zeros case, which volume aggregated per cycle is unlikely
  to be. Documented with the condition that would bring it back.
- The base also lacks `S_c`, the As-Is assignment, cycle start dates and
  capacities — in particular, without historical slot assignments the shape
  cannot be estimated at all, nor can the "volume is invariant to
  assignment" hypothesis be tested.
