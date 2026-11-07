# NEWS

## 0.3.0

- `optimize` now accepts the generic `callback` keyword. NOMAD.jl exposes no
  per-iteration hook, so the callback is invoked from the blackbox evaluator
  with a `Solution` carrying the current point and objective value.
