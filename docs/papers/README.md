# Papers

Formal write-ups (definitions, model specs, theorems/propositions) meant to
feed the team's Overleaf paper — one `.tex` file per topic/section, so two
people writing different sections don't merge-conflict in one shared file.
`formal/` formalizes in Lean any theorem/proposition/lemma that appears here.

This repo has no direct Overleaf integration: a file here is the
repo-tracked source for a section, but the Overleaf project is the actual
paper — copy or `\input` the `.tex` file into it by hand, and keep the two in
sync the same way `docs/literature/`'s `repo-note` convention keeps a Zotero
item and its deep-dive note in sync (one is the source of truth for one
thing, the other for the other — don't duplicate the write-up).

Each file should compile standalone (its own preamble) so it can be sanity
checked locally even though the shared numbering/cross-references only make
sense once it's placed inside the actual Overleaf document.

## Index

- `forecasting-volume-spec.tex` — the blocks to add to `desafio_natura.tex`
  so that its `q_{s,a,d}` parameter has a specified origin: the definition of
  $L_{s,k}$ (per-sector, per-cycle volume), the `q = L · σ` decomposition, the
  forecasting model candidates, and the comparison criterion between them.
  Each block is labelled with where it goes in the main paper; notation
  follows the main paper's (`s` sector, `d` slot, `a` day, `c` CD, so cycle
  is `k` and shape is `σ`).
