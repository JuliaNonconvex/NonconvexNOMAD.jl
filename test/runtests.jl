using NonconvexNOMAD, LinearAlgebra, Test

f(x::AbstractVector) = sqrt(x[2])
g(x::AbstractVector, a, b) = (a * x[1] + b)^3 - x[2]
x0 = [0.5, 2.3]

@testset "Alg type - $alg_type" for alg_type in [:explicit, :progressive, :custom]
    @testset "Simple constraints 1" begin
        options = NOMADOptions()
        m = Model(f)
        addvar!(m, [0.0, 0.0], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0), flags = [:explicit])
        add_ineq_constraint!(m, x -> g(x, -1, 1), flags = [:progressive])

        alg = NOMADAlg(alg_type)
        r1 = NonconvexNOMAD.optimize(m, alg, x0, options = options)
        @test abs(r1.minimum - sqrt(8 / 27)) < 1e-4
        @test norm(r1.minimizer - [1 / 3, 8 / 27]) < 1e-4

        setinteger!(m, 1, true)
        r2 = NonconvexNOMAD.optimize(m, alg, [0.0, x0[2]], options = options)
        global s2 = sum(r2.minimizer)
        @test r2.minimizer[1] - round(Int, r2.minimizer[1]) ≈ 0 atol = 1e-7

        setinteger!(m, 1, false)
        setinteger!(m, 2, true)
        r3 = NonconvexNOMAD.optimize(m, alg, [x0[1], 1.0], options = options)
        global s3 = sum(r3.minimizer)
        @test r3.minimizer[2] - round(Int, r3.minimizer[2]) ≈ 0 atol = 1e-7
    end

    @testset "Simple constraints 2" begin
        options = NOMADOptions()
        m = Model(f)
        addvar!(m, [0.0, 0.0], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0), flags = [:progressive])
        add_ineq_constraint!(m, x -> g(x, -1, 1), flags = [:explicit])

        alg = NOMADAlg(alg_type)
        r1 = NonconvexNOMAD.optimize(m, alg, x0, options = options)
        @test abs(r1.minimum - sqrt(8 / 27)) < 1e-4
        @test norm(r1.minimizer - [1 / 3, 8 / 27]) < 1e-4

        setinteger!(m, 1, true)
        r2 = NonconvexNOMAD.optimize(m, alg, [0.0, x0[2]], options = options)
        s2 = sum(r2.minimizer)
        @test r2.minimizer[1] - round(Int, r2.minimizer[1]) ≈ 0 atol = 1e-7

        setinteger!(m, 1, false)
        setinteger!(m, 2, true)
        r3 = NonconvexNOMAD.optimize(m, alg, [x0[1], 1.0], options = options)
        s3 = sum(r3.minimizer)
        @test r3.minimizer[2] - round(Int, r3.minimizer[2]) ≈ 0 atol = 1e-7
    end

    @testset "Equality constraints 1" begin
        options = NOMADOptions(linear_equality_constraints = true)
        m = Model(f)
        addvar!(m, [0.0, 0.0], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0), flags = [:explicit])
        add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)

        alg = NOMADAlg(alg_type)
        _x0 = x0 / sum(x0) * (1 / 3 + 8 / 27)
        r = NonconvexNOMAD.optimize(m, alg, _x0, options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end

    @testset "Equality constraints 2" begin
        options = NOMADOptions(linear_equality_constraints = true)
        m = Model(f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(m, x -> g(x, 2, 0), flags = [:progressive])
        add_eq_constraint!(m, x -> sum(x) - 1 / 3 - 8 / 27)

        alg = NOMADAlg(alg_type)
        _x0 = x0 / sum(x0) * (1 / 3 + 8 / 27)
        r = NonconvexNOMAD.optimize(m, alg, _x0, options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end

    @testset "Block constraints 1" begin
        options = NOMADOptions()
        m = Model(f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(
            m,
            FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2),
            flags = [:explicit],
        )

        alg = NOMADAlg(alg_type)
        r = NonconvexNOMAD.optimize(m, alg, x0, options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end

    @testset "Block constraints 2" begin
        options = NOMADOptions()
        m = Model(f)
        addvar!(m, [1e-4, 1e-4], [10.0, 10.0])
        add_ineq_constraint!(
            m,
            FunctionWrapper(x -> [g(x, 2, 0), g(x, -1, 1)], 2),
            flags = [:progressive],
        )

        alg = NOMADAlg(alg_type)
        r = NonconvexNOMAD.optimize(m, alg, x0, options = options)
        @test abs(r.minimum - sqrt(8 / 27)) < 1e-6
        @test norm(r.minimizer - [1 / 3, 8 / 27]) < 1e-6
    end
end
