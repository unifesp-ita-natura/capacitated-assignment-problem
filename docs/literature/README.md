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
