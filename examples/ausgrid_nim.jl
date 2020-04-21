using EnergyDataset, GRUtils

using Revise

using PrimalDualSDDP

const train_data = EnergyDataset.load_customer_train_data(80);
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

const nim = PrimalDualSDDP.NonIslandedModel2(Δt, capacity, 
									 		ρc, ρd, pbmax, 
									 		pbmin, pemax, Δhmax, 
						 			 		cbuy, csell, 
						 			 		permutedims([5., 10.][:,:,:,], [3,2,1]).*train_data[:,days,:],
						 			 		10)

const V = [PrimalDualSDDP.PolyhedralFunction([0. 0. 0. 0.], [-100.]) for t in 1:T]
push!(V, PrimalDualSDDP.PolyhedralFunction([-offpeak 0. 0. 0.], [0.]))

x₀s = collect(Base.product([0., 10.], [5., 10., 20.], [1.], [0.]))
const m = PrimalDualSDDP.primalsddp!(nim, V, 50, x₀s, nprune = 10);

λ₀s = collect(eachrow(V[1].λ))
const D = [PrimalDualSDDP.PolyhedralFunction([0. 0. 0. 0.], [-1e5]) for t in 1:T+1]
md = PrimalDualSDDP.dualsddp!(nim, D, 20, λ₀s, nprune = 10);
