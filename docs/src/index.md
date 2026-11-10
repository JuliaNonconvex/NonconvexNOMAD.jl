# Nonlinear optimization with the MADS (NOMAD) algorithm for continuous and discrete, constrained optimization

![JuliaNonconvex logo](https://raw.githubusercontent.com/JuliaNonconvex/NonconvexNOMAD.jl/main/docs/src/assets/logo.png)

## Description

[NOMAD.jl](https://github.com/bbopt/NOMAD.jl) is an optimization package wrapping the C++ implementation of the [NOMAD algorithm](https://dl.acm.org/doi/10.1145/1916461.1916468). `NonconvexNOMAD` allows the use of `NOMAD.jl` using the `NOMADAlg` struct. `NOMAD.jl` supports continuous and integer decision variables as well as bounds and inequality constraints. Linear equality constraints are also supported when no integer decision variables are in the model.

## Quick start

Given a model `model` and an initial solution `x0`, the following can be used to optimize the model using `NOMAD`.
```julia
using NonconvexNOMAD

alg = NOMADAlg()
options = NOMADOptions()
result = optimize(model, alg, x0, options = options)
```
`NOMAD` is an optional dependency of Nonconvex so you need to load the package to be able to use it.

## Algorithm types

There are 3 different variants of the `NOMADAlg` struct:
- `NOMADAlg(:explicit)`
- `NOMADAlg(:progressive)`
- `NOMADAlg(:custom)`

The explicit algorithm ensures all the constraints are satisfied at all times removing any infeasible point from the population. The progressive algorithm allows infeasible points to be part of the population but enforces feasibility in a progressive manner. The custom variant allows the use of flags on each constraint to declare it as `:explicit` or `:progressive`. For instance, assume `model` is the `Nonconvex` model and `g1` and `g2` are 2 constraint functions.
```julia
add_ineq_constraint!(model, g1, flags = [:explicit])
add_ineq_constraint!(model, g2, flags = [:progressive])
```
The above code declares the first constraint as explicit and the second as progressive. In other words, every point violating the first constraint will be removed from the population but the second constraint will be more progressively enforced.

## Options

The options keyword argument to the `optimize` function shown above must be an instance of the `NOMADOptions` struct when the algorihm is a `NOMADAlg`. To specify options use keyword arguments in the constructor of `NOMADOptions`, e.g:
```julia
options = NOMADOptions(max_bb_eval = 1000)
```
The following options have defaults when not specified:

- `linear_equality_constraints`: when `true`, the model's equality
  constraints are treated as linear/affine and handed to NOMAD as linear
  constraints (default `false`); see the next section
- `min_mesh_size`: minimum mesh size, a scalar or a vector with one entry
  per variable (default `0.0`)
- `initial_mesh_size`: initial mesh size, a scalar or a vector (default
  `Float64[]`, which lets NOMAD pick it)
- `granularity`: granularity for integer variables, a scalar or a vector
  (default `0.0`)
- `display_stats`: statistics displayed by NOMAD on every evaluation
  (default `["OBJ", "CONS_H", "BBE", "TIME"]`)
- `extra_display_stats`: additional statistics appended to `display_stats`
  (default `String[]`)
- `linear_constraints_atol`: tolerance on the satisfaction of the linear
  equality constraints at the initial solution (default `1e-6`)

All the other options that can be set can be found in the [`NOMAD.jl` documentation](https://bbopt.github.io/NOMAD.jl/stable/nomadProblem/).

## Equality constraints

NOMAD only supports *linear* equality constraints. When the model has
equality constraints, set the `linear_equality_constraints` option to `true`
and make sure the initial solution satisfies them, e.g:
```julia
options = NOMADOptions(linear_equality_constraints = true)
x0 = [1 / 3, 8 / 27] # sum(x0) == 1/3 + 8/27
result = optimize(model, alg, x0, options = options)
```
Without this option (or when the initial solution violates the constraint or
the model has no equality constraint at all), a descriptive `ArgumentError`
is thrown.

## Integer and binary variables

Mark variables as integer with `setinteger!`; variables whose bounds are
exactly `[0, 1]` are passed to NOMAD as binary (`B`) variables, all other
integer variables as integer (`I`) variables:
```julia
m = Model(f)
addvar!(m, [0.0, 0.0], [1.0, 10.0])
setinteger!(m, 1, true)  # binary: bounds are [0, 1]
setinteger!(m, 2, true)  # integer: bounds are [0, 10]
```

## Result

`optimize` returns a `NOMADResult` with the following fields:

- `minimizer`: the minimizer found by the algorithm (a vector of `NaN`s when
  no feasible solution was found)
- `minimum`: the minimum objective function value found (`NaN` when no
  feasible solution was found)
- `result`: the raw result of the `NOMAD.jl` solve, exposing e.g. `feasible`
  and `x_sol`
- `alg`: the `NOMADAlg` instance used
- `options`: the `NOMADOptions` instance used
