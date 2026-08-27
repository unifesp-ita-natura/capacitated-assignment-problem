# Function Design

Keep functions small and single-purpose.

* **Separate validation from construction.** Validate inputs before entering
  the function's core logic, or delegate validation to a dedicated function.
  A function that builds or transforms something should receive
  already-validated data.
* **Make state immutable.** Prefer returning new values over mutating
  variables in place. If a function needs to accumulate results, build toward
  a final return value rather than updating a shared variable across
  branches.
* **Name every distinct responsibility.** If a step inside a function
  deserves a comment explaining what it does, extract it into a named helper
  instead. The name replaces the comment and makes the top-level function
  readable as a sequence of intentions.
* **Handle branching at the boundary, not inline.** When a function behaves
  differently depending on an input type or a mode flag, dispatch to a
  dedicated function per case rather than embedding `if/else` or `switch`
  blocks inside shared logic.
