# Literature review

One file per paper/source, named `<short-slug>.md` (e.g.
`smith-2023-column-generation.md`). This keeps notes independently editable —
two people adding notes at once won't merge-conflict the way they would in
one shared running document.

## Searching

- Query the Semantic Scholar MCP tool first (`search_paper` /
  `search_paper_bulk`); fall back to WebSearch for preprints or venues it
  doesn't index (e.g. INFORMS, EURO, IPCO proceedings). Configured via
  `.mcp.json` at the repo root ([verri/semantic-scholar-mcp](https://github.com/verri/semantic-scholar-mcp))
  — set `SEMANTIC_SCHOLAR_API_KEY` in your `.env` (copy `.env.example`) for
  higher rate limits; the tool works unauthenticated too, just slower.
- Settle the search strategy before running it: keywords/synonyms,
  inclusion/exclusion criteria (date range, venue quality, method family),
  and depth (how many works, forward/backward citation depth). Write that
  strategy down (in this README or a `search-strategy.md` here) so the
  eventual Related Work section can report it and the search can be
  replicated.
- Never invent a citation. If you can't confirm a paper exists via
  Semantic Scholar, WebSearch, or a DOI, don't cite it.
- Add papers you keep to the index below as you go, not in a final batch.

## Zotero

Use the Zotero MCP tool to search, browse, and pull citation details from this
project's shared library — and to add papers/notes to it — instead of doing
that by hand. Configured via `.mcp.json` at the repo root
([zotero-mcp-server](https://pypi.org/project/zotero-mcp-server/) by
54yyyu). Key tools: `zotero_search_items`, `zotero_get_item_metadata`,
`zotero_add_by_doi` / `zotero_add_by_url` / `zotero_add_by_isbn`,
`zotero_manage_note`, `zotero_create_collection`.

Setup in your `.env` (copy `.env.example`):

- Reads work in **local mode**: open Zotero → Settings → Advanced, enable
  "Allow other applications on this computer to communicate with Zotero",
  then set `ZOTERO_LOCAL="true"`.
- **Writes always need the web API** — the local API is read-only, so
  `zotero_add_by_doi`, `zotero_manage_note`, and friends need
  `ZOTERO_API_KEY` set even when `ZOTERO_LOCAL="true"`. Create a key at
  <https://www.zotero.org/settings/keys> with write access to the group
  below.

`.env.example` already prefills `ZOTERO_LIBRARY_ID="6654414"` and
`ZOTERO_LIBRARY_TYPE="group"` for this project's shared library
([desafio_natura](https://www.zotero.org/groups/6654414/desafio_natura/library))
— that part doesn't need to change, just add your own `ZOTERO_API_KEY`.

### Collections

Two themes, three sub-collections each. File every item under a leaf
sub-collection — never leave one sitting only in the top-level theme. An item
spanning two sub-themes can go in both; don't force a false single choice.

```
Forecasting
├── Time Series Forecasting (Volume Prediction)
├── Cycle Shape Forecasting (Density Shape per Sector)
└── Anomaly Detection (INA)

Capacitated Assignment Optimization
├── Exact Methods
├── Heuristics & Metaheuristics (SA)
└── Lagrangian Relaxation
```

### Tags

Colored status tags (shared library-wide setting, `Ctrl+1`–`Ctrl+9` in the
desktop client) track triage state — the axis everyone filters on constantly:

| Tag | Color | Meaning |
|-----|-------|---------|
| `status/to-read` | red | Added, not yet triaged |
| `status/reading` | orange | Currently being read |
| `status/read-not-relevant` | yellow | Read, decided not useful — keep for record, don't cite |
| `status/core-reference` | green | Central to our work, will cite |
| `status/needs-team-review` | blue | Unsure, flag for discussion |
| `status/superseded` | purple | Replaced by a newer/better source |

Separately, use free-text `method:` tags for cross-cutting approach/technique
(e.g. `method:SA`, `method:lagrangian-relaxation`, `method:ARIMA`,
`method:LSTM`, `method:quantile-regression`) — these aren't enumerated
upfront, add them as papers need them. They're tags rather than collections
because a paper can combine multiple methods.

### Metadata & attachment rules

- **Item type**: use the correct Zotero type, not generic "Document" —
  `Journal Article`, `Conference Paper`, `Preprint`, `Thesis`, `Report` as
  applicable.
- **Required before an item counts as "processed"**: Title, Creators
  (correct order/roles), Date, DOI or URL, Abstract (pasted from source, not
  left blank).
- **Attachments**: always attach the actual PDF when accessible, not just a
  link/snapshot; use a webpage snapshot only when no PDF exists. Let Zotero
  auto-name attachments from metadata — don't hand-rename.
- **Repo cross-link**: if an item gets a deep-dive note in
  `docs/literature/<slug>.md`, put `repo-note: docs/literature/<slug>.md` in
  the Zotero item's **Extra** field. The repo file stays the source of truth
  for the actual analysis; Zotero stays the source of truth for citation
  metadata and the PDF — don't duplicate the write-up in both places.

### Other best practices

- Run Zotero's built-in **Duplicate Items** view periodically before it
  grows unmanageable.
- Consider **Better BibTeX** (desktop plugin, not API-scriptable) for stable
  citation keys once the group starts exporting a shared `.bib` — retrofitting
  keys after the fact is painful.
- PDFs get full-text indexed automatically on attach — that's what makes
  `zotero_search_items` / `zotero_get_item_fulltext` useful; don't attach
  scanned PDFs without OCR if avoidable.
- Pick one CSL citation style for the eventual paper and note it here once
  decided, so metadata pulled via the MCP tool renders consistently.
- Clear `status/to-read` on a regular cadence rather than letting it grow
  silently.
- Check Group Settings → Members in Zotero grants Read/Write to everyone who
  needs to add papers, not just the key owner.

Suggested shape for each file (adjust as needed, this isn't enforced):

```markdown
# <Full citation>

**Link/DOI:**
**Read by:**
**Date:**

## Summary
What the paper does, in a few sentences.

## Relevant to us because
Why this matters for our project specifically — not a general summary.

## Open questions / follow-ups
```

## Index

Add a one-line entry per file here as you go, so the list stays skimmable
without opening every note:

- _(add entries here)_
