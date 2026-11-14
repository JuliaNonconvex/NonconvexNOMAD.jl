@testitem "Alg type" begin
    using NonconvexNOMAD, LinearAlgebra, Test

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]
    x0 = [0.5, 2.3]

    for alg_type in [:explicit, :progressive, :custom]
        @testset "Simple constraints 1" begin
            options = NOMADOptions()
            m = Model(f)
            addvar!(m, [0.0, 0.0], [10.0, 10.0])
            add_ineq_constraint!(m, x -> g(x, 2, 0), flags=[:explicit])
            add_ineq_constraint!(m, x -> g(x, -1, 1), flags=[:progressive])

            alg = NOMADAlg(alg_type)
            r1 = NonconvexNOMAD.optimize(m, alg, x0, options=options)
            @test abs(r1.minimum - sqrt(8 / 27)) < 1e-4
            @test norm(r1.minimizer - [1 / 3, 8 / 27]) < 1e-4

            setinteger!(m, 1, true)
            r2 = NonconvexNOMAD.optimize(m, alg, [0.0, x0[2]], options=options)
            @test r2.minimizer[1] - round(Int, r2.minimizer[1]) ≈ 0 atol = 1e-7

            setinteger!(m, 1, false)
            setinteger!(m, 2, true)
            r3 = NonconvexNOMAD.optimize(m, alg, [x0[1], 1.0], options=options)
            @test r3.minimizer[2] - round(Int, r3.minimizer[2]) ≈ 0 atol = 1e-7
        end

        @testset "Simple constraints 2" begin
            options = NOMADOptions()
            m = Model(f)
            addvar!(m, [0.0, 0.0], [10.0, 10.0])
            add_ineq_constraint!(m, x -> g(x, 2, 0), flags=[:progressive])
            add_ineq_constraint!(m, x -> g(x, -1, 1), flags=[:explicit])

            alg = NOMADAlg(alg_type)
            r1 = NonconvexNOMAD.optimize(m, alg, x0, options=options)
            @test abs(r1.minimum - sqrt(8 / 27)) < 1e-4
            @test norm(r1.minimizer - [1 / 3, 8 / 27]) < 1e-4

            setinteger!(m, 1, true)
            r2 = NonconvexNOMAD.optimize(m, alg, [0.0, x0[2]], options=options)
            @test r2.minimizer[1] - round(Int, r2.minimizer[1]) ≈ 0 atol = 1e-7

            setinteger!(m, 1, false)
            setinteger!(m, 2, true)
            r3 = NonconvexNOMAD.optimize(m, alg, [x0[1], 1.0], options=options)
            @test r3.minimizer[2] - round(Int, r3.minimizer[2]) ≈ 0 atol = 1e-7
        end

        @testset "Equality constraints 1" begin
            options = NOMADOptions(linear_equality_constraints=true)
            m = Model(f)
            addvar!(m, [0.0, 0.0], [10.0, 10.0])
            add_ineq_constraint!(m, x -> g(x, 2, 0), flags=[:explicit])
            add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)

            alg = NOMADAlg(alg_type)
            _x0 = x0 / sum(x0) * (1 / 3 + 8 / 27)
            r = NonconvexNOMAD.optimize(m, alg, _x0, options=options)
            @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
            @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
        end

        @testset "Equality constraints 2" begin
            options = NOMADOptions(linear_equality_constraints=true)
            m = Model(f)
            addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
            add_ineq_constraint!(m, x -> g(x, 2, 0), flags=[:progressive])
            add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)

            alg = NOMADAlg(alg_type)
            _x0 = x0 / sum(x0) * (1 / 3 + 8 / 27)
            r = NonconvexNOMAD.optimize(m, alg, _x0, options=options)
            @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
            @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
        end

        @testset "Block constraints 1" begin
            options = NOMADOptions()
            m = Model(f)
            addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
            add_ineq_constraint!(
                m, FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2), flags=[:explicit]
            )

            alg = NOMADAlg(alg_type)
            r = NonconvexNOMAD.optimize(m, alg, x0, options=options)
            @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
            @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
        end

        @testset "Block constraints 2" begin
            options = NOMADOptions()
            m = Model(f)
            addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
            add_ineq_constraint!(
                m, FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2), flags=[:progressive]
            )

            alg = NOMADAlg(alg_type)
            r = NonconvexNOMAD.optimize(m, alg, x0, options=options)
            @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
            @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
        end
    end
end

@testitem "Equality constraint handling" begin
    using NonconvexNOMAD, LinearAlgebra, Test

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]
    x0 = [0.5, 2.3]

    # Regression test: equality constraints used to be silently dropped when
    # optimize was called without an options kwarg.
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0))
    add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)
    @test_throws ArgumentError NonconvexNOMAD.optimize(m, NOMADAlg(), x0)

    # linear_equality_constraints = true on a model without equality
    # constraints is a harmless no-op and must be accepted, not rejected.
    m2 = Model(f)
    addvar!(m2, [0.0, 0.0], [10.0, 10.0])
    r = NonconvexNOMAD.optimize(
        m2, NOMADAlg(), x0, options=NOMADOptions(linear_equality_constraints=true)
    )
    @test isfinite(r.minimum)
    @test 0.0 <= r.minimum <= sqrt(2.3)
end

@testitem "Mesh-size option validation" begin
    using NonconvexNOMAD, LinearAlgebra, Test

    # Regression test: NOMAD.jl used to detect per-variable mesh-size option
    # length mismatches only deep inside its problem constructor with an
    # unhelpful assertion. The workspace constructor must validate the
    # lengths up front with an informative ArgumentError.
    f(x::AbstractVector) = sqrt(x[2])
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    vm, vx0, _ = NonconvexCore.tovecmodel(m, [0.5, 2.3])

    for (name, bad_value) in
        [(:min_mesh_size, [1e-4]), (:initial_mesh_size, [1.0]), (:granularity, [1.0])]
        options = NOMADOptions(; (name => bad_value,)...)
        err = try
            Workspace(vm, NOMADAlg(), vx0; options=options)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(string(name), err.msg)
        @test occursin("length 1", err.msg)
        @test occursin("2 input variables", err.msg)
    end

    # Scalar values and correctly sized vectors are still accepted.
    for options in [
        NOMADOptions(
            min_mesh_size=[1e-4, 1e-4], initial_mesh_size=[1.0, 1.0], granularity=[0.0, 0.0]
        ),
        NOMADOptions(min_mesh_size=1e-4, initial_mesh_size=1.0, granularity=0.0),
        NOMADOptions(),  # initial_mesh_size defaults to the empty vector (NOMAD default)
    ]
        @test Workspace(vm, NOMADAlg(), vx0; options=options) isa
            NonconvexNOMAD.NOMADWorkspace
    end
end

@testitem "finite_or_inf" begin
    using NonconvexNOMAD, Test

    @test NonconvexNOMAD.finite_or_inf(NaN) == Inf
    @test NonconvexNOMAD.finite_or_inf(-Inf) == -Inf
    @test NonconvexNOMAD.finite_or_inf(Inf) == Inf
    @test NonconvexNOMAD.finite_or_inf(1.5) == 1.5
end

@testitem "nomad_result" begin
    using NonconvexNOMAD, LinearAlgebra, Test

    f(x::AbstractVector) = sqrt(x[2])
    x0 = [0.5, 2.3]

    # Both branches of the NOMAD.jl 3.0 result handling: feasible solutions
    # expose `x_sol`, failed runs return NaN results.
    alg = NOMADAlg()
    options = NOMADOptions()
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    r1 = NonconvexNOMAD.nomad_result(
        (feasible=true, x_sol=[1.0, 2.0], status=0), x0, sum, alg, options
    )
    @test r1.minimum == 3.0
    @test r1.minimizer == [1.0, 2.0]
    @test r1.result.feasible

    r2 = NonconvexNOMAD.nomad_result(
        (feasible=false, x_sol=nothing, status=-2), x0, sum, alg, options
    )
    @test isnan(r2.minimum)
    @test all(isnan, r2.minimizer)
    @test !r2.result.feasible

    r3 = NonconvexNOMAD.nomad_result(
        (feasible=true, x_sol=nothing, status=1), x0, sum, alg, options
    )
    @test isnan(r3.minimum)
    @test all(isnan, r3.minimizer)
end

@testitem "Iteration callback" begin
    using NonconvexNOMAD, LinearAlgebra, Test

    f(x::AbstractVector) = sqrt(x[2])
    g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]
    x0 = [0.5, 2.3]

    options = NOMADOptions()
    m = Model(f)
    addvar!(m, [0.0, 0.0], [10.0, 10.0])
    add_ineq_constraint!(m, x -> g(x, 2, 0), flags=[:explicit])
    add_ineq_constraint!(m, x -> g(x, -1, 1), flags=[:progressive])

    iterates = Any[]
    r = NonconvexNOMAD.optimize(
        m,
        NOMADAlg(:explicit),
        x0,
        options=options,
        callback=s -> push!(iterates, (x=copy(s.x), f=s.f)),
    )
    @test length(iterates) > 0
    @test all(s -> isfinite(s.f), iterates)
    @test abs(r.minimum - sqrt(8 / 27)) < 1e-4
end

@testitem "solver capabilities" begin
    using NonconvexNOMAD, Test

    caps = solver_capabilities(NOMADAlg())
    @test caps isa ConstraintCapabilities
    @test caps.linear_eq === true
    @test caps.nonlinear_eq === false  # only linear equalities (option)
    @test caps.linear_ineq === true
    @test caps.nonlinear_ineq === true
    @test caps.linear_sd === false
    @test caps.nonlinear_sd === false
    @test caps.box === true
    @test caps.integer === true
    @test caps.infinite_bounds === true
end
