abstract type DecisionHazardModel end

function solve!(::DecisionHazardModel, m::JuMP.Model, xₜ::Vector{Float64})
	fix.(m[:xₜ], xₜ)
	optimize!(m)
	@assert termination_status(m) == MOI.OPTIMAL
	return
end

function new_cut!(dhm::DecisionHazardModel, Vₜ::PolyhedralFunction, 
				  m::JuMP.Model, xₜ::Vector{Float64})
	solve!(dhm, m, xₜ)
	λ = dual.(FixRef.(m[:xₜ]))
	γ = objective_value(m) - λ'*xₜ
	Vₜ.λ = cat(Vₜ.λ, λ', dims = 1)
	push!(Vₜ.γ, γ)
	return
end

function update!(::DecisionHazardModel, mₜ::JuMP.Model, Vₜ₊₁::PolyhedralFunction)
	nξ = length(mₜ[:θ])
	for ξ in 1:nξ
		@constraint(mₜ, mₜ[:θ][ξ] >= Vₜ₊₁.λ[end,:]'*mₜ[:xₜ₊₁][ξ] + Vₜ₊₁.γ[end])
	end
	return
end

function control!(dhm::DecisionHazardModel, m::JuMP.Model, xₜ::Vector{Float64})
	solve!(dhm, m, xₜ)
	return value.(m[:uₜ])
end

function forward_pass(dhm::DecisionHazardModel,
					  m::Vector{JuMP.Model}, 
					  ξscenarios::Array{Float64, 3},
					  x₀::Vector{Float64},
					  fₜ::Function)
	T = size(ξscenarios,1)
	n_pass = size(ξscenarios,2)
	xscenarios = fill(0., T+1, n_pass, length(x₀))
	for pass in 1:n_pass
		xₜ = x₀	
		xscenarios[1,pass,:] .= x₀
		for (t, ξₜ₊₁) in enumerate(eachrow(ξscenarios[:,pass,:]))
			uₜ = control!(dhm, m[t], xₜ)
			xₜ = fₜ(t, xₜ, uₜ, ξₜ₊₁)
			xscenarios[t+1,pass,:] .= xₜ
		end
	end
	return xscenarios
end

function backward_pass!(dhm::DecisionHazardModel,
						m::Vector{JuMP.Model},
					  	V::Vector{PolyhedralFunction},
					  	xscenarios::Array{Float64,3})
	T = length(m)
	n_pass = size(xscenarios, 2)
	for pass in 1:n_pass
		new_cut!(dhm, V[T], m[T], xscenarios[T,pass,:])
	end
	for t in T:-1:1
		update!(dhm, m[t], V[t+1])
		for pass in 1:n_pass
			new_cut!(dhm, V[t], m[t], xscenarios[t,pass,:])
		end
	end
	return 
end

function sddp!(dhm::DecisionHazardModel, 
			   V::Array{PolyhedralFunction}, 
			   n_pass::Int, 
			   x₀s::Array)
	T, S = size(dhm.ξs)
	m = [bellman_operator(dhm, t, Vₜ₊₁) for (t, Vₜ₊₁) in enumerate(V[2:end])]
	println("Bellman JuMP Models initialized")
	println("Now running $(n_pass) sddp passes")
	@showprogress for i in 1:n_pass
		ξscenarios = dhm.ξs[:,rand(1:S,1),:]
		x₀ = [rand(x₀s)...]
		xscenarios = forward_pass(dhm, m, ξscenarios, x₀, dhm.fₜ)
		backward_pass!(dhm, m, V, xscenarios)
	end
	return m
end

function upper_bound(dhm::DecisionHazardModel,
					 m::Vector{JuMP.Model}, 
					 ξscenarios::Array{Float64, 3},
					 x₀s::Vector{Float64},
					 fₜ::Function,
					 montecarlo_sample_size::Int = 1000)
	x₀ = [rand(x₀s)...]
	xscenarios = forward_pass(dhm, m, ξscenarios, x₀, dhm.fₜ)
end
