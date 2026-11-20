# Algorithm types

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
