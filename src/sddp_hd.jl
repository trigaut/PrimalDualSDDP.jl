abstract type HazardDecisionModel end

function solve!(::HazardDecisionModel, 
				m::JuMP.Model, 
				xₜ::Vector{Float64}, 
				wₜ::Vector{Float64})
	fix.(m[:xₜ], xₜ)
	fix.(m[:wₜ], wₜ)
	optimize!(m)
	@assert termination_status(m) == MOI.OPTIMAL
	return
end

function new_cut!(hdm::HazardDecisionModel, 
				  Vₜ::PolyhedralFunction, 
				  m::JuMP.Model, 
				  xₜ::Vector{Float64})
	λ = fill(0., length(m[:xₜ]))
	γ = 0.
	for (iw, wₜ) in enumerate(eachrow(hdm.ξ[t]))
		solve!(hdm, m, xₜ, wₜ)
	 	λ += hdm.πξ[t][iw] * dual.(FixRef.(m[:xₜ]))
		γ += hdm.πξ[t][iw] * objective_value(m) - λ'*xₜ
	end
	Vₜ.λ = cat(Vₜ.λ, λ', dims = 1)
	push!(Vₜ.γ, γ)
	return
end

function update!(::HazardDecisionModel, 
				 mₜ::JuMP.Model, 
				 Vₜ₊₁::PolyhedralFunction)
	nξ = length(mₜ[:θ])
	@constraint(mₜ, mₜ[:θ] >= Vₜ₊₁.λ[end,:]'*mₜ[:xₜ₊₁] + Vₜ₊₁.γ[end])
	return
end

function control!(hdm::HazardDecisionModel, m::JuMP.Model, 
				  xₜ::Vector{Float64}, wₜ::Vector{Float64})
	solve!(hdm, m, xₜ, wₜ)
	return value.(m[:uₜ])
end

function forward_pass(hdm::HazardDecisionModel,
					  m::Vector{JuMP.Model}, 
					  ξscenario::Vector{Float64},
					  x₀::Vector{Float64},
					  fₜ::Function)
	T = length(ξscenario)
	xscenario = fill(0., T+1, length(x₀))
	xscenario[1,:] .= x₀
	xₜ = x₀	
	for (t, ξₜ) in enumerate(ξscenario)
		uₜ = control!(hdm, m[t], xₜ, ξₜ)
		xₜ = fₜ(t, xₜ, uₜ, [ξₜ])
		xscenario[t+1,:] .= xₜ
	end
	return xscenario
end

function backward_pass!(hdm::HazardDecisionModel,
						m::Vector{JuMP.Model},
					  	V::Vector{PolyhedralFunction},
					  	xscenario::Array{Float64,2})
	T = length(m)
	for t in T:-1:1
		if t < T
			update!(hdm, m[t], V[t+1])
		end
		xₜ = xscenario[t,:]
		new_cut!(hdm, V[t], m[t], xₜ)
	end
	return 
end

function sddp!(hdm::HazardDecisionModel, 
			   V::Array{PolyhedralFunction}, 
			   n_pass::Int, 
			   x₀s::Array)
	T, S = size(hdm.ϵs)
	m = [bellman_operator(hdm, t, Vₜ₊₁) for (t, Vₜ₊₁) in enumerate(V[2:end])]
	println("Bellman JuMP Models initialized")
	println("Now running $(n_pass) sddp passes")
	@showprogress for i in 1:n_pass
		ϵ_scenario = hdm.ϵs[:,rand(1:S)]
		x₀ = [rand(x₀s)...]
		xscenario = forward_pass(hdm, m, ϵ_scenario, x₀, hdm.fₜ)
		backward_pass!(hdm, m, V, xscenario)
	end
	return m
end
