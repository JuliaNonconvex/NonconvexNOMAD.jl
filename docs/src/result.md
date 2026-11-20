# Result

`optimize` returns a `NOMADResult` with the following fields:

- `minimizer`: the minimizer found by the algorithm (a vector of `NaN`s when
  no feasible solution was found)
- `minimum`: the minimum objective function value found (`NaN` when no
  feasible solution was found)
- `result`: the raw result of the `NOMAD.jl` solve, exposing e.g. `feasible`
  and `x_sol`
- `alg`: the `NOMADAlg` instance used
- `options`: the `NOMADOptions` instance used
