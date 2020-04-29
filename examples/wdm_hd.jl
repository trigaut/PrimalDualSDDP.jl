using Random

using Revise

using PrimalDualSDDP

const T = 24
const S = 365
const Δt = 1.
const cap = 10.
const umax = cap/2
const bins = 10

Random.seed!(1234)
const csell = rand(T)
const rain_scenarios = 1.2*cap .* rand(T,S)

const wdm = PrimalDualSDDP.WaterDamModel(Δt, cap, 
                                         umax, csell, 
                                         rain_scenarios, 
                                         bins)


const V = [PrimalDualSDDP.PolyhedralFunction([0.]', [-1e4]) for t in 1:T]
push!(V, PrimalDualSDDP.PolyhedralFunction([0.]', [0.]))
x₀s = [(0.,)]
const m = PrimalDualSDDP.primalsddp!(wdm, V, 100, x₀s, nprune = 10);

l1_regularization = maximum(PrimalDualSDDP.lipschitz_constant.(V))

const D = [PrimalDualSDDP.PolyhedralFunction([0.]', [-1e4]) for t in 1:T]
push!(D, PrimalDualSDDP.δ([0.], 1e4))

const λ₀s = collect(eachrow(V[1].λ))
const md = PrimalDualSDDP.dualsddp!(wdm, D, 100, λ₀s, 
                                    nprune = 10, 
                                    l1_regularization = l1_regularization);

const Vinner = PrimalDualSDDP.PolyhedralFenchelTransform.(D, l1_regularization)
