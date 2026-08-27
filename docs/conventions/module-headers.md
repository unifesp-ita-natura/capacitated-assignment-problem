# Module Header Convention

Every tracked Python file starts with a short module docstring before
`from __future__ import annotations` or any imports.

Preferred format:

```python
"""One sentence describing the file's job."""
```

Header rules:

* Keep the header to a single sentence whenever possible.
* Update the header when a file's public purpose changes.
* Avoid YAML-style metadata blocks in module docstrings.
* Put detailed architecture/workflow notes in `docs/`, not in per-file headers.

This is the convention followed across the codebase, e.g.
`src/solver/example.py`, `src/config/loader.py`.
