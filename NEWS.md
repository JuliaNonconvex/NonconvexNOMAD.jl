# NEWS

## 0.4.5

- Docs: package-name section titles and left-aligned logos (no code changes).
## 0.4.4

- The linear-equality matrix is formed with `NonconvexCore.value_jacobian`
  (JuMP-converted models get the exact evaluator Jacobian); the direct
  Zygote dependency is dropped.

## 0.4.3

- Added docstrings for `NOMADAlg` and `NOMADOptions` (REPL help and API reference).


## 0.4.2

- Validated the NOMAD mesh-size options and made the equality-constraint
  path a documented no-op; widened the dependency compatibility bounds;
  scheduled formatter workflow.

## 0.4.1

- Widened the Zygote compat for the Core-2 ecosystem; added the
  organization logo; docs workflow fixes (stable dispatch, gh-pages
  persistence, destination fallback); covered every exported feature in
  the docs and tests.

## 0.4.0

- NonconvexCore 2 with strict solver capability declarations
  (`infinite_bounds`); DocumenterFragments docs; tests migrated to
  TestItems with JET/Aqua/formatting quality gates; Julia 1.10+.

## 0.3.0

- `optimize` now accepts the generic `callback` keyword. NOMAD.jl exposes no
  per-iteration hook, so the callback is invoked from the blackbox evaluator
  with a `Solution` carrying the current point and objective value.
