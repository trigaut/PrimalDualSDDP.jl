using EnergyDataset, GRUtils

using Revise

using OlenEff

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

const nim = OlenEff.NonIslandedModel(Δt, capacity, 
									 ρc, ρd, pbmax, 
									 pbmin, pemax, Δhmax, 
						 			 cbuy, csell, 
						 			 5 .* train_data[:,days,1] .- 10 .* train_data[:,days,2], 10)

const V = [OlenEff.PolyhedralFunction([0. 0. 0.], [-100.]) for t in 1:T]
push!(V, OlenEff.PolyhedralFunction([-offpeak 0. 0.], [0.]))
	
x₀s = collect(Base.product([0.], [ 20.], [0.]))
const m = OlenEff.sddp!(nim, V, 20, x₀s)
