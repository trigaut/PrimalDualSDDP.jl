include(joinpath(@__DIR__,"models", "nim_net_hd.jl"))

using JLD, GRUtils

#const train_data = EnergyDataset.load_customer_train_data(80);
begin const 
train_data = load(joinpath(@__DIR__,"ausgrid_train_80.jld"))["data"]
T = size(train_data,1)
Δt = 24/T
S = size(train_data,2)
days = 150:180
capacity = 10.
ρc = 0.96
ρd = 0.95
pbmax = 5.
pbmin = -5.
pemax = 100.
Δhmax = 20.
csell = fill(0., T)

# Australian time of use tariff
peak = 0.2485
shoulder = 0.0644
offpeak = 0.0255
cbuy = cat(fill(offpeak, 7*2), 
              fill(shoulder, 10*2), 
              fill(peak, 4*2), 
              fill(shoulder, 1*2), 
              fill(offpeak, 2*2), 
              dims=1);
end

const primal_pruner = PrimalDualSDDP.ExactPruner(SOLVER, 
                                                 lb = [0., 0., Pmax], 
                                                 ub = [capacity, Δhmax, Dmax])

const nim = NonIslandedNetHDModel(Δt, capacity, 
                                  ρc, ρd, pbmax, 
                                  pbmin, pemax, Δhmax, 
                                  cbuy, csell, 
                                  5 .* train_data[:,days,1] .- 10 .* train_data[:,days,2], 10)

const primal_pruner = PrimalDualSDDP.ExactPruner(SOLVER, 
                                                 lb = [0., 0., Pmax], 
                                                 ub = [capacity, Δhmax, Dmax])

const V = [PrimalDualSDDP.PolyhedralFunction([0. 0.], [-100.]) for t in 1:T]
push!(V, PrimalDualSDDP.PolyhedralFunction([-offpeak 0.], [0.]))
    
x₀s = collect(Base.product([0., 10.], [5., 10., 20.]))
const m = PrimalDualSDDP.primalsddp!(nim, V, 50, x₀s, 
                                     nprune = 10, pruner = primal_pruner)

#const l1_regularization = maximum(PrimalDualSDDP.lipschitz_constant.(V))
const l1_regularization = 162


const dual_pruner = PrimalDualSDDP.ExactPruner(SOLVER)
const λ₀s = PrimalDualSDDP.FixedInitSampler(collect(eachrow(V[1].λ)))
const D = [PrimalDualSDDP.PolyhedralFunction([0. 0.], [-1e10]) for t in 1:T]
push!(D, PrimalDualSDDP.δ([-offpeak, 0.], 1e3))
const md = PrimalDualSDDP.dualsddp!(nim, D, 100, λ₀s, 
                                    nprune = 25, 
                                    l1_regularization = l1_regularization)


const Vinner = PrimalDualSDDP.PolyhedralFenchelTransform.(D, l1_regularization, 
                                                          nprune = 10, 
                                                          pruner = dual_pruner)
