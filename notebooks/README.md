# notebooks

Exploratory Jupyter notebooks. Keep these read-only/throwaway — once an
analysis is worth keeping, move the logic into `src/` (or `scripts/`) with
tests, and leave the notebook as a thin driver over that code.

`.ipynb_checkpoints/` is gitignored. The pre-commit hook runs `nbstripout`
on staged `.ipynb` files automatically, so cell outputs never end up in git
history — you don't need to clear them by hand before committing.
