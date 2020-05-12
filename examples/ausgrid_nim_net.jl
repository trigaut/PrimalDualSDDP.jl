using JLD, GRUtils

include(joinpath(@__DIR__,"models", "nim_net_dh.jl"))

#const train_data = EnergyDataset.load_customer_train_data(80);
const train_data = load(joinpath(@__DIR__,"ausgrid_train_80.jld"))["data"]
const T = size(train_data,1)
const Δt = 24/T
const S = size(train_data,2)
const days = 150:180
const capacity = 10.
const ρc = 0.96
const ρd = 0.95
const pbmax = 5.
const pbmin = -5.
const pemax = 100.
const Δhmax = 20.
const csell = fill(0., T)

# Australian time of use tariff
const peak = 0.2485
const shoulder = 0.0644
const offpeak = 0.0255
const cbuy = cat(fill(offpeak, 7*2), 
                 fill(shoulder, 10*2), 
                 fill(peak, 4*2), 
                 fill(shoulder, 1*2), 
                 fill(offpeak, 2*2), 
                 dims=1);

const nim = NonIslandedNetModel(Δt, capacity, 
                               ρc, ρd, pbmax, 
                               pbmin, pemax, Δhmax, 
                               cbuy, csell, 
                               5 .* train_data[:,days,1] .- 
                               10 .* train_data[:,days,2], 
                               10)

const V = [PrimalDualSDDP.PolyhedralFunction([0., 0., 0.] => -100.) for t in 1:T]
push!(V, PrimalDualSDDP.PolyhedralFunction([-offpeak, 0., 0.] => 0.))
for v in V
  PrimalDualSDDP.update_bounds!(v, 1, lower_bound = 0., upper_bound = capacity)
  PrimalDualSDDP.update_bounds!(v, 2, lower_bound = 0., upper_bound = 2*capacity)
  PrimalDualSDDP.update_bounds!(v, 3, lower_bound = -1e3, upper_bound = 1e3)
end
    
const x₀s = collect(Base.product([0., 10.], [5., 10., 20.], [0.]))
m = PrimalDualSDDP.primalsddp!(nim, V, 200, x₀s, nprune = 50, prunetol = 1e-2)

# const λ₀s = collect(eachrow(V[1].λ))
# const D = [PrimalDualSDDP.PolyhedralFunction([0. 0. 0.], [-1e10]) for t in 1:T]
# push!(D, PrimalDualSDDP.δ([-offpeak, 0., 0.], 1e3))

# md = PrimalDualSDDP.dualsddp!(nim, D, 200, λ₀s, nprune = 50)
