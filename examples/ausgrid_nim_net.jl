using JLD, GRUtils

include(joinpath(@__DIR__,"models", "nim_net_dh.jl"))

#const train_data = EnergyDataset.load_customer_train_data(80);
begin const
full_train_data = load(joinpath(@__DIR__,"ausgrid_train_80.jld"))["data"]
days = 150:180
train_data = 5 .* full_train_data[:,days,1] .- 10 .* full_train_data[:,days,2]
Pmax, Dmax = extrema(train_data)

T = size(train_data,1)
Δt = 24/T
S = size(train_data,2)
capacity = 10.
ρc = 0.96
ρd = 0.95
pbmax = 5.
pbmin = -5.
pemax = 100.
Δhmax = 2*capacity
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

const nim = NonIslandedNetModel(Δt, capacity, 
                                ρc, ρd, pbmax, 
                                pbmin, pemax, Δhmax, 
                                cbuy, csell, 
                                train_data, 
                                10)


const primal_pruner = PrimalDualSDDP.ExactPruner(SOLVER, 
                                                 lb = [0., 0., Pmax], 
                                                 ub = [capacity, Δhmax, Dmax])

const V = [PrimalDualSDDP.PolyhedralFunction([0. 0. 0.], [-100.]) for t in 1:T]
push!(V, PrimalDualSDDP.PolyhedralFunction([-offpeak 0. 0.], [0.]))
    
const x₀s = collect(Base.product([0., 10.], [5., 10., 20.], [0.]))
m = PrimalDualSDDP.primalsddp!(nim, V, 200, x₀s, nprune=50, pruner=primal_pruner)

const λ₀s = PrimalDualSDDP.FixedInitSampler(collect(eachrow(V[1].λ)))
const D = [PrimalDualSDDP.PolyhedralFunction([0. 0. 0.], [-1e10]) for t in 1:T]
push!(D, PrimalDualSDDP.δ([-offpeak, 0., 0.], 1e3))

const l1_regularization = 200.

const dual_pruner = PrimalDualSDDP.ExactPruner(SOLVER, 
                                               lb = [0., 0., Pmax], 
                                               ub = [capacity, 2*capacity, Dmax])

md = PrimalDualSDDP.dualsddp!(nim, D, 200, λ₀s, 
                              nprune=50, pruner=dual_pruner,
                              l1_regularization=l1_regularization)
