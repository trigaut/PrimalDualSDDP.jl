abstract type DecisionHazardModel end

abstract type HazardDecisionModel end

abstract type DecisionHazardDecisionModel end


function solve!(m::JuMP.Model, xₜ::Vector{Float64})
	fix.(m[:xₜ], xₜ)
	optimize!(m)
	@assert termination_status(m) == MOI.OPTIMAL
	return
end

function new_cut!(Vₜ::PolyhedralFunction, m::JuMP.Model, xₜ::Vector{Float64})
	solve!(m, xₜ)
	λ = dual.(FixRef.(m[:xₜ]))
	γ = objective_value(m) - λ'*xₜ
	Vₜ.λ = cat(Vₜ.λ, λ', dims = 1)
	push!(Vₜ.γ, γ)
	return
end

function update!(mₜ::JuMP.Model, Vₜ₊₁::PolyhedralFunction)
	nξ = length(mₜ[:θ])
	for ξ in 1:nξ
		@constraint(mₜ, mₜ[:θ][ξ] >= Vₜ₊₁.λ[end,:]'*mₜ[:xₜ₊₁][ξ] + Vₜ₊₁.γ[end])
	end
	return
end

function control!(m::JuMP.Model, xₜ::Vector{Float64})
	solve!(m, xₜ)
	return value.(m[:uₜ])
end

function forward_pass(m::Vector{JuMP.Model}, 
					  ξscenario::Vector{Float64},
					  x₀::Vector{Float64},
					  fₜ::Function)
	T = length(ξscenario)
	xscenario = fill(0., T+1, length(x₀))
	xscenario[1,:] .= x₀
	xₜ = x₀	
	for (t, ξₜ₊₁) in enumerate(ξscenario)
		uₜ = control!(m[t], xₜ)
		xₜ = fₜ(t, xₜ, uₜ, [ξₜ₊₁])
		xscenario[t+1,:] .= xₜ
	end
	return xscenario
end

function backward_pass!(m::Vector{JuMP.Model},
					  	V::Vector{PolyhedralFunction},
					  	xscenario::Array{Float64,2})
	T = length(m)
	for t in T:-1:1
		if t < T
			update!(m[t], V[t+1])
		end
		xₜ = xscenario[t,:]
		new_cut!(V[t], m[t], xₜ)
	end
	return 
end

function sddp!(dhm::DecisionHazardModel, 
			   V::Array{PolyhedralFunction}, 
			   n_pass::Int, 
			   x₀s::Array)
	T, S = size(dhm.ϵs)
	m = [bellman_operator(dhm, t, Vₜ₊₁) for (t, Vₜ₊₁) in enumerate(V[2:end])]
	println("Bellman JuMP Models initialized")
	println("Now running $(n_pass) sddp passes")
	@showprogress for i in 1:n_pass
		ϵ_scenario = dhm.ϵs[:,rand(1:S)]
		x₀ = [rand(x₀s)...]
		xscenario = forward_pass(m, ϵ_scenario, x₀, dhm.fₜ)
		backward_pass!(m, V, xscenario)
	end
	return m
end