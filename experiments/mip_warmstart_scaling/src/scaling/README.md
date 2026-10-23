# `src/scaling`

Generic machinery for sweeping instance size (e.g. sectors x cycles) while
solving each point, shared by any scaling experiment.

- `sweep.py` — `sweep(cycles, sectors, run_point, max_timeouts, completed, on_row)`
  grows sectors per cycle count and moves to the next cycle count after
  `max_timeouts` timeouts (`maxTimeLimit`/`noSolution`) or when the data runs
  out of sectors. `run_point` is injected and returns a `Trial`, so the logic
  needs no solver to test.
- `store.py` — append-only CSV: `append_row` writes each solved point
  immediately, `load_completed` reads them back as the `completed` argument,
  which makes a sweep resumable (skipped points still count toward the stop rules).

See `experiments/mip_warmstart_scaling/` for use.
