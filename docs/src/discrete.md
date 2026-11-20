# Integer and binary variables

Mark variables as integer with `setinteger!`; variables whose bounds are
exactly `[0, 1]` are passed to NOMAD as binary (`B`) variables, all other
integer variables as integer (`I`) variables:
```julia
m = Model(f)
addvar!(m, [0.0, 0.0], [1.0, 10.0])
setinteger!(m, 1, true)  # binary: bounds are [0, 1]
setinteger!(m, 2, true)  # integer: bounds are [0, 10]
```
