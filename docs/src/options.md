# Options

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
