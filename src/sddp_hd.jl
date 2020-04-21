abstract type HazardDecisionModel end

function initialize_lift_primal!(m::JuMP.Model,
						  		 hdm::HazardDecisionModel,
					      		 t::Int,
					      		 Vₜ₊₁::PolyhedralFunction)
	
	@variable(m, θ)
	for (λ, γ) in eachcut(Vₜ₊₁)
		@constraint(m, θ >= λ'*m[:xₜ₊₁] + γ)	
	end
	obj_expr = objective_function(m)
	@objective(m, Min, obj_expr + θ)
	return
end

function initialize_lift_dual!(m::JuMP.Model,
						  	   hdm::HazardDecisionModel,
					      	   t::Int,
					      	   Dₜ₊₁::PolyhedralFunction)
	nξ = length(m[:μₜ₊₁])
	@variable(m, θ[1:nξ])
	for (λ, γ) in eachcut(Dₜ₊₁)
		for i in 1:nξ
			@constraint(m, θ[i] >= λ'*m[:μₜ₊₁][i] + γ)
		end
	end
	obj_expr = objective_function(m)
	@objective(m, Min, obj_expr + sum(dhm.πξ[t][i]*θ[i] for i in 1:nξ))
	return
end

function primalsolve!(::HazardDecisionModel,
					   m::JuMP.Model, 
					   xₜ::Vector{Float64},
					   ξₜ₊₁::Vector{Float64})
	fix.(m[:xₜ], xₜ, force=true)
	fix.(m[:ξₜ₊₁], ξₜ₊₁, force=true)
	optimize!(m)
	@assert termination_status(m) == MOI.OPTIMAL println(m)
	return
end

function dualsolve!(::HazardDecisionModel,
					m::JuMP.Model, 
					μₜ::Vector{Float64})
	fix.(m[:μₜ], μₜ, force=true)
	optimize!(m)
	@assert termination_status(m) == MOI.OPTIMAL println(m)
	return
end

function new_cut!(hdm::HazardDecisionModel, 
				  Vₜ::PolyhedralFunction, 
				  m::JuMP.Model, 
				  xₜ::Vector{Float64})
	ξₜ₊₁ = hdm.ξ[t]
	πₜ₊₁ = hdm.πξ[t]
	λ = 0.
	γ = 0.
	for (i, πᵢ) in enumerate(πₜ₊₁)
		primalsolve!(hdm, m, xₜ, ξₜ₊₁[i,:])
		λ += πᵢ.*dual.(FixRef.(m[:xₜ]))
		γ += πᵢ.*(objective_value(m) - λ'*xₜ)
	end
	Vₜ.λ = cat(Vₜ.λ, λ', dims = 1)
	push!(Vₜ.γ, γ)
	return
end

function dual_new_cut!(hdm::HazardDecisionModel, 
				  	   Dₜ::PolyhedralFunction, 
				  	   m::JuMP.Model, 
				  	   μₜ::Vector{Float64})
	dualsolve!(dhm, m, μₜ)
	λ = dual.(FixRef.(m[:μₜ]))
	γ = objective_value(m) - λ'*μₜ
	Dₜ.λ = cat(Dₜ.λ, λ', dims = 1)
	push!(Dₜ.γ, γ)
	return
end

function update!(hdm::HazardDecisionModel, 
				 mₜ::JuMP.Model, 
				 Vₜ₊₁::PolyhedralFunction)
	nξ = length(mₜ[:θ])
	@constraint(mₜ, mₜ[:θ] >= Vₜ₊₁.λ[end,:]'*mₜ[:xₜ₊₁] + Vₜ₊₁.γ[end])
	return
end

function dualupdate!(hdm::HazardDecisionModel, 
				 	 mₜ::JuMP.Model, 
				 	 Dₜ₊₁::PolyhedralFunction)
	nξ = length(mₜ[:θ])
	for i in 1:nξ
		@constraint(mₜ, mₜ[:θ][i] >= Dₜ₊₁.λ[end,:]'*mₜ[:μₜ₊₁][i] + Dₜ₊₁.γ[end])
	end
	return
end

function control!(hdm::HazardDecisionModel, 
				  m::JuMP.Model, 
				  xₜ::Vector{Float64},
				  ξₜ₊₁::Vector{Float64})
	primalsolve!(hdm, m, xₜ, ξₜ₊₁)
	return value.(m[:uₜ])
end

function dualstate!(hdm::HazardDecisionModel, 
				  	m::JuMP.Model, 
				  	μₜ::Vector{Float64})
	dualsolve!(dhm, m, μₜ)
	return map(μ♯ -> value.(μ♯), m[:μₜ₊₁])
end


function forward_pass(hdm::HazardDecisionModel,
					  m::Vector{JuMP.Model}, 
					  ξscenarios::Array{Float64, 3},
					  x₀::Vector{Float64})
	T = size(ξscenarios,1)
	n_pass = size(ξscenarios,2)
	xscenarios = fill(0., T+1, n_pass, length(x₀))
	for pass in 1:n_pass
		xₜ = x₀	
		xscenarios[1,pass,:] .= x₀
		for (t, ξₜ₊₁) in enumerate(eachrow(ξscenarios[:,pass,:]))
			uₜ₊₁ = control!(hdm, m[t], xₜ, ξₜ₊₁)
			xₜ = dhm.fₜ(t, xₜ, uₜ₊₁, ξₜ₊₁)
			xscenarios[t+1,pass,:] .= xₜ
		end
	end
	return xscenarios
end

function backward_pass!(hdm::HazardDecisionModel,
						m::Vector{JuMP.Model},
					  	V::Vector{PolyhedralFunction},
					  	xscenarios::Array{Float64,3})
	T = length(m)
	n_pass = size(xscenarios, 2)
	for pass in 1:n_pass
		new_cut!(hdm, V[T], m[T], xscenarios[T,pass,:])
	end
	for t in T:-1:1
		update!(hdm, m[t], V[t+1])
		for pass in 1:n_pass
			new_cut!(hdm, V[t], m[t], xscenarios[t,pass,:])
		end
	end
	return 
end

function dual_forward_pass(hdm::HazardDecisionModel,
					  	   m::Vector{JuMP.Model}, 
					  	   ξscenarios::Array{Float64, 3},
					  	   μ₀::Vector{Float64})
	T = size(ξscenarios,1)
	n_pass = size(ξscenarios,2)
	μscenarios = fill(0., T+1, n_pass, length(μ₀))
	for pass in 1:n_pass
		μₜ = μ₀	
		μscenarios[1,pass,:] .= μ₀
		for (t, ξₜ₊₁) in enumerate(eachrow(ξscenarios[:,pass,:]))
			μₜ₊₁ = dualstate!(dhm, m[t], μₜ)
			μscenarios[t+1,pass,:] .= rand(μₜ₊₁)
			μₜ = μscenarios[t+1,pass,:]
		end
	end
	return μscenarios
end


function dual_backward_pass!(hdm::HazardDecisionModel,
							 m::Vector{JuMP.Model},
					  		 D::Vector{PolyhedralFunction},
					  		 μscenarios::Array{Float64,3})
	T = length(m)
	n_pass = size(μscenarios, 2)
	for pass in 1:n_pass
		dual_new_cut!(dhm, D[T], m[T], μscenarios[T,pass,:])
	end
	for t in T:-1:1
		dualupdate!(dhm, m[t], D[t+1])
		for pass in 1:n_pass
			dual_new_cut!(dhm, D[t], m[t], μscenarios[t,pass,:])
		end
	end
	return 
end


function primalsddp!(hdm::HazardDecisionModel, 
					 V::Array{PolyhedralFunction}, 
					 n_pass::Int, 
					 x₀s::Array;
					 nprune = n_pass)
	println("** Primal SDDP, in Hazard Decision , with $(n_pass) passes and $(div(n_pass,nprune)) pruning  **")
	T, S = size(hdm.ξs)
	m = [bellman_operator(hdm, t) for t in 1:T]
	for (t, Vₜ₊₁) in enumerate(V[2:end])
		initialize_lift_primal!(m[t], hdm, t, Vₜ₊₁)
	end
	println("Primal Bellman JuMP Models initialized")
	println("Now running sddp passes")
	@showprogress for i in 1:n_pass
		ξscenarios = hdm.ξs[:,rand(1:S,1),:]
		x₀ = [rand(x₀s)...]
		xscenarios = forward_pass(hdm, m, ξscenarios, x₀)
		backward_pass!(hdm, m, V, xscenarios)
		if mod(i,nprune) == 0
			println("\n Performing pruning number $(div(i,nprune))")
			V[1] = unique(V[1])
			for (t, Vₜ₊₁) in enumerate(V[2:end])
				V[t+1] = unique(Vₜ₊₁)
				m[t] = bellman_operator(hdm, t)
				initialize_lift_primal!(m[t], hdm, t, V[t+1])
			end
		end
	end
	return m
end

function dualsddp!(hdm::HazardDecisionModel, 
			   		D::Array{PolyhedralFunction}, 
			   		n_pass::Int, 
			   		μ₀s::Array;
			   		nprune = n_pass)
	println("** Dual SDDP with $(n_pass) passes, $(div(n_pass,nprune)) pruning  **")
	T, S = size(hdm.ξs)
	m = [dual_bellman_operator(hdm, t) for t in 1:T]
	for (t, Dₜ₊₁) in enumerate(D[2:end])
		initialize_lift_dual!(m[t], hdm, t, Dₜ₊₁)
	end
	println("Dual Bellman JuMP Models initialized")
	println("Now running sddp passes")
	@showprogress for i in 1:n_pass
		ξscenarios = hdm.ξs[:,rand(1:S,1),:]
		μ₀ = [rand(μ₀s)...]
		μscenarios = dual_forward_pass(hdm, m, ξscenarios, μ₀)
		dual_backward_pass!(hdm, m, D, μscenarios)
		if mod(i,nprune) == 0
			println("\n Performing pruning number $(div(i,nprune))")
			D[1] = unique(D[1])
			for (t, Dₜ₊₁) in enumerate(D[2:end])
				D[t+1] = unique(Dₜ₊₁)
				m[t] = dual_bellman_operator(hdm, t)
				initialize_lift_dual!(m[t], hdm, t, D[t+1])
			end
		end
	end
	return m
end

