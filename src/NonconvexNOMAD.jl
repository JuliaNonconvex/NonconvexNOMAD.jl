module NonconvexNOMAD

export NOMADAlg, NOMADOptions

using Reexport, Parameters, Zygote
@reexport using NonconvexCore
using NonconvexCore: VecModel, AbstractResult
using NonconvexCore: AbstractOptimizer, CountingFunction
import NonconvexCore: optimize, optimize!, Workspace
import NOMAD: NOMAD

struct NOMADAlg <: AbstractOptimizer
    type::Symbol
end
NOMADAlg() = NOMADAlg(:explicit) # :progressive, :custom

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
    linear_equality_constraints = false,
    min_mesh_size = 0.0,
    initial_mesh_size = Float64[],
    granularity = 0.0,
    display_stats = ["OBJ", "CONS_H", "BBE", "TIME"],
    extra_display_stats = String[],
    linear_constraints_atol = 1e-6,
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

mutable struct NOMADWorkspace{M<:VecModel,X<:AbstractVector,O<:NOMADOptions,A<:NOMADAlg} <:
               Workspace
    model::M
    x0::X
    options::O
    alg::A
end
function NOMADWorkspace(
    model::VecModel,
    optimizer::NOMADAlg,
    x0::AbstractVector = getinit(model);
    options = NOMADOptions(),
    kwargs...,
)
    return NOMADWorkspace(model, copy(x0), options, optimizer)
end
struct NOMADResult{M1,M2,R,A,O} <: AbstractResult
    minimizer::M1
    minimum::M2
    result::R
    alg::A
    options::O
end

function NonconvexCore._optimize_precheck(
    model::NonconvexCore.AbstractModel,
    ::NOMADAlg,
    x0;
    options,
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
    return
end

@generated function drop_ks(nt::NamedTuple{names}, ::Val{ks}) where {names,ks}
    ns = Tuple(setdiff(names, ks))
    return :(NamedTuple{$ns}(nt))
end

finite_or_inf(x::T) where {T} = isfinite(x) ? x : convert(T, Inf)

function optimize!(workspace::NOMADWorkspace)
    @unpack model, options, x0, alg = workspace
    if options.nt.linear_equality_constraints
        @assert length(model.eq_constraints.fs) > 0
        A = Zygote.jacobian(model.eq_constraints, x0)[1]
        b = -model.eq_constraints(zeros(length(x0)))
    else
        A = nothing
        b = nothing
    end

    if A !== nothing && norm(A * x0 - b) > options.nt.linear_constraints_atol
        throw(
            ArgumentError(
                "The initial solution doesn't satisfy the linear equality constraints.",
            ),
        )
    end
    if length(model.ineq_constraints.fs) > 0 && any(>(0), model.ineq_constraints(x0))
        throw(
            ArgumentError(
                "The initial solution doesn't satisfy the inequality constraints.",
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
            :linear_equality_constraints,
            :min_mesh_size,
            :initial_mesh_size,
            :granularity,
        )),
    )

    for k in keys(nomad_options)
        setproperty!(nomad_problem.options, k, nomad_options[k])
    end

    result = NOMAD.solve(nomad_problem, x0)
    minimizer = result.x_best_feas
    if minimizer === nothing
        return NOMADResult(fill(NaN, length(x0)), NaN, result, alg, options)
    else
        return NOMADResult(minimizer, obj(minimizer), result, alg, options)
    end
end

function Workspace(model::VecModel, optimizer::NOMADAlg, args...; kwargs...)
    return NOMADWorkspace(model, optimizer, args...; kwargs...)
end

end
