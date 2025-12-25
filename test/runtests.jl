using Genqo
using Test

@testset "SPDC" begin
    @test !any(isnan, spdc.spin_density_matrix(1e-4, 0.9, 0.6, [1,0,1,1,0,0,1,0]))
    @test !isnan(spdc.probability_success(1e-2, 0.9))
end

@testset "ZALM" begin
    @test !any(isnan, zalm.spin_density_matrix(1e-4, 0.9, 0.6, 0.8, [1,0,1,1,0,0,1,0]))
    @test !isnan(zalm.probability_success(1e-2, 0.8, 0.6, 0.9, 0.2))
    @test !isnan(zalm.fidelity(1e-2, 0.8, 0.6, 0.9))
end
