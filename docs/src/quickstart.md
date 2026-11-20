# Quick start

Given a model `model` and an initial solution `x0`, the following can be used to optimize the model using `NOMAD`.
```julia
using NonconvexNOMAD

alg = NOMADAlg()
options = NOMADOptions()
result = optimize(model, alg, x0, options = options)
```
`NOMAD` is an optional dependency of Nonconvex so you need to load the package to be able to use it.
