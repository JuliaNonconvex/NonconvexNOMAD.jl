# Equality constraints

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
