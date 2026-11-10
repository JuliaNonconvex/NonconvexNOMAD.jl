@testitem "JET" begin
    using JET, NonconvexNOMAD
    JET.test_package(NonconvexNOMAD; target_modules=(NonconvexNOMAD,))
end
