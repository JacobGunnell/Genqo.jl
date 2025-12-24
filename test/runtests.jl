using Genqo
using Test

@testset "ZALM" begin
    @test_nowarn zalm.spin_density_matrix(1e-4, 0.9, 0.6, 0.8, [1,0,1,1,0,0,1,0])
    @test_nowarn zalm.probability_success(1e-2, 0.8, 0.6, 0.9, 0.2)
    @test_nowarn zalm.fidelity(1e-2, 0.8, 0.6, 0.9)
end
