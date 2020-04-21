mutable struct WaterDamModel <: HazardDecisionModel
	Δt::Float64
	capacity::Float64
	umax::Float64
	csell::Vector{Float64} # csell < cbuy or it won't work !
	ξ
	πξ
	ξs
	fₜ
end

function WaterDamModel(Δt::Float64, capacity::Float64, 
					   umax::Float64, csell::Vector{Float64}, 
					   waterfall_scenarios::Array{Float64,2}, 
					   bins::Int)
	
	ξ, πξ = discrete_white_noise(waterfall_scenarios, bins);

	function fₜ(t, xₜ, uₜ₊₁, ξₜ₊₁)
		return [xₜ[1] - uₜ₊₁[1] + ξₜ₊₁[1] ]
	end

	WaterDamModel(Δt, capacity umax, csell, 
				  ξ, πξ, waterfall_scenarios[:,:,:], fₜ)
end

