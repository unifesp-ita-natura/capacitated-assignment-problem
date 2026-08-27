# Formal proofs (Lean 4 / Mathlib)

Lean 4 formalizations of the structural theorems, propositions, and lemmas in
`docs/papers/`. Each file formalizes exactly one numbered result and states,
in its module docstring, which `\theorem`/`\proposition`/`\lemma` label it
corresponds to.

If this project doesn't need formal proofs, delete this directory and drop
the matching bullet from `AGENTS.md`.

## Layout

```
formal/
├── lakefile.toml     # Lake project file, depends on Mathlib — rename "Project" to your project name
├── lean-toolchain    # pinned Lean toolchain (matches the Mathlib rev above)
└── Project/
    └── Example.lean  # stub — replace with real theorem files, one per result
```

## Building

Requires [elan](https://github.com/leanprover/elan) (the Lean toolchain
manager) and network access to GitHub and Mathlib's Azure blob cache.

```bash
cd formal
elan --version || curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
lake exe cache get   # download precompiled Mathlib .olean files (~a few GB)
lake build           # compiles Project/*.lean
```

`lean-toolchain` pins the exact `leanprover/lean4` release that
`lakefile.toml`'s Mathlib revision was built against, so `lake exe cache get`
finds a matching prebuilt cache instead of rebuilding Mathlib from source.

## Gotcha: import order

Lean requires `import` statements before any other top-level content,
including the file's module docstring (`/-! ... -/`). Put imports first,
then the docstring, then declarations. If a pinned Lean/Mathlib toolchain is
tracking a recent release candidate, Mathlib API drift (renamed lemmas,
restructured files into directories) can surface between updates; if
`lake build` fails after bumping the toolchain or Mathlib rev, check for
renamed identifiers before assuming the proof itself is wrong.
