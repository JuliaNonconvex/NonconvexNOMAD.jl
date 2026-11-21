# NonconvexNOMAD

![JuliaNonconvex logo](https://raw.githubusercontent.com/JuliaNonconvex/NonconvexNOMAD.jl/main/docs/src/assets/logo.png)

## Description

[NOMAD.jl](https://github.com/bbopt/NOMAD.jl) is an optimization package wrapping the C++ implementation of the [NOMAD algorithm](https://dl.acm.org/doi/10.1145/1916461.1916468). `NonconvexNOMAD` allows the use of `NOMAD.jl` using the `NOMADAlg` struct. `NOMAD.jl` supports continuous and integer decision variables as well as bounds and inequality constraints. Linear equality constraints are also supported when no integer decision variables are in the model.

See [Quick start](quickstart.md) for the first steps, [Algorithm
types](algorithms.md) for the available MADS variants, [Options](options.md)
for the algorithm options, [Equality constraints](equalities.md) and
[Integer and binary variables](discrete.md) for the model requirements,
[Result](result.md) for the fields of the returned result and
[Docstrings](docstrings.md) for the full API reference.
