module NonconvexNOMAD

export NOMADAlg, NOMADOptions

using Reexport, Parameters, Zygote
@reexport using NonconvexCore
using NonconvexCore: VecModel, AbstractResult
using NonconvexCore: AbstractOptimizer, CountingFunction
using NonconvexCore: Solution, ConvergenceState, NoCallback
import NonconvexCore: optimize, optimize!, Workspace
import NOMAD: NOMAD

struct NOMADAlg <: AbstractOptimizer
    type::Symbol
end
NOMADAlg() = NOMADAlg(:explicit) # :progressive, :custom

# Nonlinear equality constraints are supported only when the
# `linear_equality_constraints` option is set (validated with a descriptive
# error by `_optimize_precheck`); integer variables are supported via NOMAD's
# granularity/input types.
function NonconvexCore.solver_capabilities(::NOMADAlg)
    return ConstraintCapabilities(;
        linear_eq=true,
        nonlinear_eq=false,  # only with the linear_equality_constraints option
        linear_ineq=true,
        nonlinear_ineq=true,
        linear_sd=false,
        nonlinear_sd=false,
        box=true,
        continuous=true,
        binary=true,
        integer=true,
        infinite_bounds=true,
    )
end

struct NOMADOptions{N<:NamedTuple}
    nt::N
end

#= Other options include:
max_bb_eval::Int=20000, max_sgte_eval::Int=1000, opportunistic_eval::Bool=true, use_cache::Bool=true, random_eval_sort::Bool=false, lh_search::Tuple{Int, Int}=(0, 0),
quad_model_search::Bool=true, sgtelib_search::Bool=false, sgtelib_model_trials::Int=1,
speculative_search::Bool=true, speculative_search_max::Int=1, nm_search::Bool=true,
nm_search_stop_on_success::Bool=false, max_time::Union{Nothing,Int}=nothing,
linear_converter::String=SVD
=#
function NOMADOptions(;
    linear_equality_constraints=false,
    min_mesh_size=0.0,
    initial_mesh_size=Float64[],
    granularity=0.0,
    display_stats=["OBJ", "CONS_H", "BBE", "TIME"],
    extra_display_stats=String[],
    linear_constraints_atol=1e-6,
    kwargs...,
)
    display_stats = unique(vcat(display_stats, extra_display_stats))
    return NOMADOptions(
        merge(
            (;
                linear_equality_constraints,
                min_mesh_size,
                initial_mesh_size,
                granularity,
                display_stats,
                linear_constraints_atol,
            ),
            NamedTuple(kwargs),
        ),
    )
end

mutable struct NOMADWorkspace{
    M<:VecModel,X<:AbstractVector,O<:NOMADOptions,A<:NOMADAlg,F<:Function
} <: Workspace
    model::M
    x0::X
    options::O
    alg::A
    callback::F
end
function NOMADWorkspace(
    model::VecModel,
    optimizer::NOMADAlg,
    x0::AbstractVector=NonconvexCore.getinit(model);
    options=NOMADOptions(),
    callback::Function=NoCallback(),
    kwargs...,
)
    validate_mesh_size_options(options, length(x0))
    return NOMADWorkspace(model, copy(x0), options, optimizer, callback)
end

"""
    validate_mesh_size_options(options, nb_inputs)

Checks that the per-variable mesh-size options (`min_mesh_size`,
`initial_mesh_size` and `granularity`) passed as vectors have one entry per
input variable. NOMAD.jl only detects length mismatches deep inside its
problem constructor with an unhelpful assertion, so they are validated up
front here with an informative error. Scalar values are broadcast to all
variables in `optimize!`; an empty `initial_mesh_size` vector is
valid because NOMAD.jl derives a default initial mesh size from the
granularity.
"""
function validate_mesh_size_options(options::NOMADOptions, nb_inputs::Integer)
    for name in (:min_mesh_size, :initial_mesh_size, :granularity)
        val = getproperty(options.nt, name)
        if val isa AbstractVector
            name === :initial_mesh_size && isempty(val) && continue
            length(val) == nb_inputs || throw(
                ArgumentError(
                    "The `$(name)` option is a vector of length $(length(val)) but the model has $(nb_inputs) input variables. Pass a vector with one value per variable, or a scalar value to use it for all variables.",
                ),
            )
        end
    end
    return nothing
end
struct NOMADResult{M1,M2,R,A,O} <: AbstractResult
    minimizer::M1
    minimum::M2
    result::R
    alg::A
    options::O
end

function NonconvexCore._optimize_precheck(
    model::NonconvexCore.AbstractModel, ::NOMADAlg, x0; options=NOMADOptions(), kwargs...
)
    length(model.eq_constraints.fs) == 0 ||
        options.nt.linear_equality_constraints ||
        throw(
            ArgumentError(
                "NOMAD does not support nonlinear equality constraints, only bound constraints, inequality constraints and linear equality constraints. You can set the `linear_equality_constraints` option to `true` if the equality constraint functions are indeed linear/affine.",
            ),
        )
    length(model.sd_constraints.fs) == 0 || throw(
        ArgumentError(
            "NOMAD does not support semidefinite constraints, only bound constraints, inequality constraints and linear equality constraints.",
        ),
    )
    return nothing
end

@generated function drop_ks(nt::NamedTuple{names}, ::Val{ks}) where {names,ks}
    ns = Tuple(setdiff(names, ks))
    return :(NamedTuple{$ns}(nt))
end

finite_or_inf(x::T) where {T} = isnan(x) ? convert(T, Inf) : x

function optimize!(workspace::NOMADWorkspace)
    @unpack model, options, x0, alg, callback = workspace
    if options.nt.linear_equality_constraints && length(model.eq_constraints.fs) > 0
        A = Zygote.jacobian(model.eq_constraints, x0)[1]
        b = -model.eq_constraints(zeros(length(x0)))
    else
        # `linear_equality_constraints = true` on a model without equality
        # constraints is a harmless no-op and is accepted.
        A = nothing
        b = nothing
        length(model.eq_constraints.fs) == 0 || throw(
            ArgumentError(
                "NOMAD does not support nonlinear equality constraints, only bound constraints, inequality constraints and linear equality constraints. You can set the `linear_equality_constraints` option to `true` if the equality constraint functions are indeed linear/affine.",
            ),
        )
    end

    if A !== nothing && norm(A * x0 - b) > options.nt.linear_constraints_atol
        throw(
            ArgumentError(
                "The initial solution doesn't satisfy the linear equality constraints."
            ),
        )
    end
    if length(model.ineq_constraints.fs) > 0 && any(>(0), model.ineq_constraints(x0))
        throw(
            ArgumentError(
                "The initial solution doesn't satisfy the inequality constraints."
            ),
        )
    end

    nb_outputs = 1
    if length(model.ineq_constraints.fs) > 0
        if alg.type == :explicit
            N = length(model.ineq_constraints(x0))
            nb_outputs += N
            ineqT = fill("EB", N)
        elseif alg.type == :progressive
            N = length(model.ineq_constraints(x0))
            nb_outputs += N
            ineqT = fill("PB", N)
        elseif alg.type == :custom
            ineqT = mapreduce(vcat, model.ineq_constraints.fs) do f
                if (:explicit in f.flags)
                    N = length(f(x0))
                    nb_outputs += N
                    fill("EB", N)
                elseif (:progressive in f.flags)
                    N = length(f(x0))
                    nb_outputs += N
                    fill("PB", N)
                else
                    throw(
                        ArgumentError(
                            """Unsupported flag `"type"` value, please choose from `:explicit` and `:progressive`.""",
                        ),
                    )
                end
            end
        else
            throw(ArgumentError("Unsupported NOMAD algorithm type."))
        end
    else
        ineqT = String[]
    end
    output_types = [["OBJ"]; ineqT]

    obj = getobjective(model)
    obj === nothing && throw(ArgumentError("The model has no objective function."))
    obj(x0)
    model.ineq_constraints(x0)

    eval_bb =
        x -> begin
            try
                if length(model.ineq_constraints.fs) > 0
                    out = [finite_or_inf(obj(x)); finite_or_inf.(model.ineq_constraints(x))]
                else
                    out = [finite_or_inf(obj(x))]
                end
                callback(nomad_solution(x, out[1]))
                return (true, true, out)
            catch err
                if !(err isa DomainError)
                    rethrow(err)
                end
                out = fill(Inf, nb_outputs)
                return (false, true, out)
            end
        end

    nb_inputs = length(x0)
    input_types = map(enumerate(model.integer)) do (i, int)
        if int && getmin(model, i) == 0 && getmax(model, i) == 1
            return "B"
        elseif int
            return "I"
        else
            return "R"
        end
    end

    min_mesh_size = if options.nt.min_mesh_size isa Real
        fill(options.nt.min_mesh_size, nb_inputs)
    else
        options.nt.min_mesh_size
    end

    initial_mesh_size = if options.nt.initial_mesh_size isa Real
        fill(options.nt.initial_mesh_size, nb_inputs)
    else
        options.nt.initial_mesh_size
    end

    granularity = if options.nt.granularity isa Real
        fill(options.nt.granularity, nb_inputs)
    else
        options.nt.granularity
    end

    lower_bound = getmin(model)
    upper_bound = getmax(model)

    nomad_problem = NOMAD.NomadProblem(
        nb_inputs,
        nb_outputs,
        output_types,
        eval_bb;
        input_types,
        lower_bound,
        upper_bound,
        A,
        b,
        min_mesh_size,
        initial_mesh_size,
        granularity,
    )

    nomad_options = drop_ks(
        options.nt,
        Val((
            :linear_equality_constraints, :min_mesh_size, :initial_mesh_size, :granularity
        )),
    )

    for k in keys(nomad_options)
        setproperty!(nomad_problem.options, k, nomad_options[k])
    end

    result = NOMAD.solve(nomad_problem, x0)
    return nomad_result(result, x0, obj, alg, options)
end

# NOMAD.jl 3.0 returns a NamedTuple; feasible runs expose `x_sol`, while
# failed runs only signal infeasibility. Keep the old semantics: NaN result
# when no feasible minimizer was found.
function nomad_result(result, x0, obj, alg, options)
    if result.feasible && result.x_sol !== nothing
        minimizer = result.x_sol
        return NOMADResult(minimizer, obj(minimizer), result, alg, options)
    else
        return NOMADResult(fill(NaN, length(x0)), NaN, result, alg, options)
    end
end

function Workspace(model::VecModel, optimizer::NOMADAlg, args...; kwargs...)
    return NOMADWorkspace(model, optimizer, args...; kwargs...)
end

"""
    nomad_solution(x, f)

Builds the `Solution` passed to the Nonconvex `callback` from a blackbox
evaluation of NOMAD at `x` with objective value `f`.
"""
function nomad_solution(x::AbstractVector, f::Real)
    convstate = ConvergenceState{Float64}()
    return Solution(
        copy(x),
        copy(x),
        Float64[],
        f,
        f,
        zeros(length(x)),
        Float64[],
        zeros(0, length(x)),
        convstate,
    )
end

end
