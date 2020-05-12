push!(LOAD_PATH, "..")
include(joinpath(@__DIR__,"models", "water_dam_hd.jl"))

using Random
Random.seed!(1234)

begin const
    T = 24
    S = 365
    Δt = 1.0
    cap = 10.0
    umax = cap/2
    bins = 10
    csell = rand(T)
    rain_scenarios = 1.2 * cap .* rand(T,S)
end

const wdm = WaterDamModel(Δt, cap,
                          umax, csell,
                          rain_scenarios,
                          bins)

const primal_pruner = PrimalDualSDDP.ExactPruner(SOLVER, lb = [0.], ub = [capacity])

V = [PrimalDualSDDP.PolyhedralFunction([0.0]', [-1e4]) for t in 1:T]
push!(V, PrimalDualSDDP.PolyhedralFunction([0.0]', [0.0]))
x₀s = [(0.0,)]
primal_models = @time PrimalDualSDDP.primalsddp!(wdm, V, 100, x₀s, 
                                                 pruner=primal_pruner, 
                                                 nprune=200);

l1_regularization = maximum(PrimalDualSDDP.lipschitz_constant.(V))

D = [PrimalDualSDDP.PolyhedralFunction([0.0]', [-1e4]) for t in 1:T]
push!(D, PrimalDualSDDP.δ([0.0], 1e4))


const dual_pruner = PrimalDualSDDP.ExactPruner(SOLVER)

λ₀s = collect(eachrow(V[1].λ))
dual_models = @time PrimalDualSDDP.dualsddp!(wdm, D, 100, λ₀s,
                                             nprune=101,
                                             pruner=dual_pruner,
                                             l1_regularization=l1_regularization);

Vinner = PrimalDualSDDP.PolyhedralFenchelTransform.(D, l1_regularization);
