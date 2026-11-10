# Additional feature coverage for the exported NOMADAlg / NOMADOptions API.
# These @testitem blocks exercise paths that the main suite does not cover:
# NOMADOptions defaults and merging, NOMADAlg type variants, NOMADResult
# fields, binary variables, the DomainError fallback inside the blackbox
# evaluator, and the descriptive error paths in the Workspace/optimize!

@testitem "features: NOMADOptions defaults and merging" begin
    using NonconvexNOMAD, Test

    # --- defaults ---
    o = NOMADOptions()
    @test o isa NonconvexNOMAD.NOMADOptions
    @test o.nt.linear_equality_constraints === false
    @test o.nt.min_mesh_size == 0.0
    @test o.nt.initial_mesh_size == Float64[]
    @test o.nt.granularity == 0.0
    @test o.nt.display_stats == ["OBJ", "CONS_H", "BBE", "TIME"]
    @test o.nt.linear_constraints_atol == 1e-6

    # --- extra_display_stats are appended and deduplicated ---
    o2 = NOMADOptions(extra_display_stats=["BBE", "TIME", "ITER"])
    @test o2.nt.display_stats == ["OBJ", "CONS_H", "BBE", "TIME", "ITER"]
    @test o2.nt.linear_equality_constraints === false

    # --- arbitrary NOMAD options are accepted as keyword arguments ---
    o3 = NOMADOptions(max_bb_eval=1000, linear_equality_constraints=true)
    @test o3.nt.max_bb_eval == 1000
    @test o3.nt.linear_equality_constraints === true
    @test o3.nt.display_stats == ["OBJ", "CONS_H", "BBE", "TIME"]

    # --- per-variable (vector) mesh/granularity values are kept as-is ---
    o4 = NOMADOptions(
        min_mesh_size=[1e-4, 1e-4], initial_mesh_size=[1.0, 1.0], granularity=[1, 1]
    )
    @test o4.nt.min_mesh_size == [1e-4, 1e-4]
    @test o4.nt.initial_mesh_size == [1.0, 1.0]
    @test o4.nt.granularity == [1, 1]
end

@testitem "features: NOMADAlg type variants" begin
    using NonconvexNOMAD, Test

    a = NOMADAlg()
    @test a isa NonconvexNOMAD.NOMADAlg
    @test a.type === :explicit
    @test NOMADAlg(:explicit).type === :explicit
    @test NOMADAlg(:progressive).type === :progressive
    @test NOMADAlg(:custom).type === :custom
end

@testitem "features: NOMADResult fields" begin
    using NonconvexNOMAD, LinearAlgebra, Test

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0), flags=[:explicit])
    add_ineq_constraint!(m, x -> g(x, -1, 1), flags=[:progressive])

    alg = NOMADAlg(:explicit)
    options = NOMADOptions()
    r = NonconvexNOMAD.optimize(m, alg, [0.5, 2.3], options=options)
    @test r isa NonconvexNOMAD.NOMADResult
    @test r.minimizer isa AbstractVector
    @test r.minimum isa Real
    @test r.result isa NamedTuple
    @test r.result.feasible
    @test r.alg === alg
    @test r.options === options
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-4
end

@testitem "features: binary variables" begin
    using NonconvexNOMAD, LinearAlgebra, Test

    # Variables with bounds exactly [0, 1] are declared as NOMAD binary ("B")
    # input types; other integer variables use the "I" input type.
    m = Model(x -> sum(abs2, x .- [0.0, 1.0]))
    addvar!(m, [0.0, 0.0], [1.0, 10.0])
    setinteger!(m, 1, true)  # binary
    setinteger!(m, 2, true)  # integer
    r = NonconvexNOMAD.optimize(
        m, NOMADAlg(), [0.0, 5.0], options=NOMADOptions(max_bb_eval=5000)
    )
    @test r.minimizer[1] - round(Int, r.minimizer[1]) ≈ 0 atol = 1e-7
    @test r.minimizer[2] - round(Int, r.minimizer[2]) ≈ 0 atol = 1e-7
    @test abs(r.minimizer[1] - 0.0) < 1e-7
    @test abs(r.minimizer[2] - 1.0) < 1e-7
    @test r.minimum < 1e-8
end

@testitem "features: DomainError evaluations are treated as infeasible" begin
    using NonconvexNOMAD, LinearAlgebra, Test

    # Points outside the lower-left box throw a DomainError; the blackbox
    # evaluator must swallow it and report the point as infeasible.
    f(x::AbstractVector) =
        x[1] < 0.5 && x[2] < 0.5 ? sum(abs2, x) : throw(DomainError(x, "out of domain"))
    m = Model(f)
    addvar!(m, [0.0, 0.0], [1.0, 1.0])
    r = NonconvexNOMAD.optimize(
        m, NOMADAlg(), [0.1, 0.1], options=NOMADOptions(max_bb_eval=5000)
    )
    @test isfinite(r.minimum)
    @test r.minimum < 1e-2
    @test norm(r.minimizer) < 0.2
end

@testitem "features: error paths" begin
    using NonconvexNOMAD, LinearAlgebra, Test

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]

    # --- semidefinite constraints are rejected ---
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_sd_constraint!(m, x -> [x[1] 0.0; 0.0 x[2]], 2)
    err = try
        NonconvexNOMAD.optimize(m, NOMADAlg(), [0.5, 2.3])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("semidefinite", err.msg)

    # --- initial solution violating the linear equality constraints ---
    m2 = Model(f)
    addvar!(m2, [0.0, 0.0], [10.0, 10.0])
    add_eq_constraint!(m2, x -> sum(x) - 1 / 3 - 8 / 27)
    err = try
        NonconvexNOMAD.optimize(
            m2,
            NOMADAlg(),
            [0.5, 2.3];
            options=NOMADOptions(linear_equality_constraints=true),
        )
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("linear equality constraints", err.msg)

    # --- initial solution violating the inequality constraints ---
    m3 = Model(f)
    addvar!(m3, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m3, x -> x[1] + x[2] - 2.0) # violated at [0.5, 2.3]
    err = try
        NonconvexNOMAD.optimize(m3, NOMADAlg(), [0.5, 2.3])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("inequality constraints", err.msg)

    # --- :custom with a constraint lacking :explicit/:progressive flags ---
    m4 = Model(f)
    addvar!(m4, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m4, x -> g(x, 2, 0)) # no flags
    err = try
        NonconvexNOMAD.optimize(m4, NOMADAlg(:custom), [0.5, 2.3])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Unsupported flag", err.msg)

    # --- unknown algorithm type with inequality constraints ---
    m5 = Model(f)
    addvar!(m5, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m5, x -> g(x, 2, 0))
    err = try
        NonconvexNOMAD.optimize(m5, NOMADAlg(:bogus), [0.5, 2.3])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Unsupported NOMAD algorithm type", err.msg)
end

@testitem "features: custom flags and per-variable options through optimize" begin
    using NonconvexNOMAD, LinearAlgebra, Test

    # The :custom variant resolves each constraint's own flags, and the
    # per-variable mesh options are forwarded to NOMAD.
    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0), flags=[:explicit])
    add_ineq_constraint!(m, x -> g(x, -1, 1), flags=[:progressive])
    options = NOMADOptions(
        max_bb_eval=3000,
        min_mesh_size=[1e-8, 1e-8],
        initial_mesh_size=[1.0, 1.0],
        granularity=[0.0, 0.0],
    )
    r = NonconvexNOMAD.optimize(m, NOMADAlg(:custom), [0.5, 2.3], options=options)
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-4
    @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-4
end
