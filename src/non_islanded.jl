mutable struct PolyhedralFunction
	λ::Array{Float64,2}
	γ::Array{Float64,1}
end

ncuts(V::PolyhedralFunction) = size(V.λ, 1)
dim(V::PolyhedralFunction) = size(V.λ, 2)
eachcut(V::PolyhedralFunction) = zip(eachrow(V.λ), V.γ)

function PolyhedralFunction()
	return PolyhedralFunction(Array{Float64,2}(undef, 0,0), Vector{Float64}())
end

struct NonIslandedModel
	Δt::Float64
	capacity::Float64
	ρc::Float64
	ρd::Float64
	pbmax::Float64
	pbmin::Float64
	pemax::Float64
	Δhmax::Float64
	cbuy::Vector{Float64}
	csell::Vector{Float64} # csell < cbuy or it won't work !
	α
	β
	ϵ
	πϵ
	ϵs
	fₜ
end

function NonIslandedModel(Δt::Float64, capacity::Float64, 
						  ρc::Float64, ρd::Float64, 
						  pbmax::Float64, pbmin::Float64, 
						  pemax::Float64, Δhmax::Float64, 
						  cbuy::Vector{Float64}, csell::Vector{Float64}, 
						  net_d_scenarios::Array{Float64,2}, bins::Int)
	
	α1, β1, _ = fitar(net_d_scenarios)
	
	ϵs1 = net_d_scenarios[2:end,:] .- α1 .* net_d_scenarios[1:end-1,:] .- β1

	α = cat([0.], α1, dims = 1)[:,1]
	β = cat([0.], β1, dims = 1)
	ϵs = cat(net_d_scenarios[1:1,:], ϵs1, dims = 1)

	ϵ, πϵ = discrete_white_noise(ϵs, bins);

	function fₜ(t, xₜ, uₜ, ξₜ₊₁)
		xₜ₊₁ = [xₜ[1] + ρc*uₜ[1] - 1/ρd*uₜ[2],
				xₜ[2] - uₜ[1] - uₜ[2],
				α[t]*xₜ[3] + β[t] + ξₜ₊₁[1] ]
		return xₜ₊₁
	end

	NonIslandedModel(Δt, capacity, ρc, ρd, 
					 pbmax, pbmin, pemax, 
					 Δhmax, cbuy, csell, 
					 α, β, ϵ, πϵ, ϵs, fₜ)
end

function bellman_operator(nim::NonIslandedModel, t::Int, Vₜ₊₁::PolyhedralFunction)

	m = JuMP.Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))
	
	@variable(m, 0. <= uₜ⁺ <= nim.pbmax)
	@variable(m, 0. <= uₜ⁻ <= -nim.pbmin)

	@variable(m, socₜ)
	@expression(m, socₜ₊₁, socₜ + nim.ρc * uₜ⁺ - 1/nim.ρd * uₜ⁻)
	@constraint(m, 0. <= socₜ₊₁ <= nim.capacity)

	@variable(m, hₜ)
	@expression(m, hₜ₊₁, hₜ - uₜ⁺ - uₜ⁻)
	@constraint(m, hₜ₊₁ >= 0.)

	@variable(m, net_dₜ)
	ϵ = nim.ϵ[t]
	nξ = length(ϵ)
	@expression(m, net_dₜ₊₁[ξ=1:nξ], ϵ[ξ] + nim.α[t] * net_dₜ + nim.β[t])

	@expression(m, importₜ₊₁[ξ=1:nξ], net_dₜ₊₁[ξ] + uₜ⁺ - uₜ⁻)
	@variable(m, ebuy[1:nξ] >= 0.)
	@constraint(m, ebuy .>= importₜ₊₁)
	if nim.pemax < Inf
		@constraint(m, ebuy  .<= nim.pemax)
	end
	@expression(m, esell[ξ=1:nξ], importₜ₊₁[ξ] - ebuy[ξ])

	@variable(m, θ[1:nξ])
	for (λ, γ) in eachcut(Vₜ₊₁)
		for ξ in 1:nξ
			@constraint(m, θ[ξ] >= λ'*[socₜ₊₁, hₜ₊₁, net_dₜ₊₁[ξ]] + γ)
		end
	end
	@objective(m, Min, sum(nim.πϵ[t][ξ]*(nim.cbuy[t]*ebuy[ξ] + nim.csell[t]*esell[ξ] + θ[ξ]) for ξ in 1:nξ))

	@expression(m, xₜ, [socₜ, hₜ, net_dₜ])
	@expression(m, xₜ₊₁[ξ=1:nξ], [socₜ₊₁, hₜ₊₁, net_dₜ₊₁[ξ]])
	@expression(m, uₜ, [uₜ⁺, uₜ⁻])

	return m
end

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

function sddp!(nim::NonIslandedModel, V::Array{PolyhedralFunction}, 
			   n_pass::Int, x₀s::Array)
	T, S = size(nim.ϵs)
	m = [bellman_operator(nim, t, Vₜ₊₁) for (t, Vₜ₊₁) in enumerate(V[2:end])]
	println("Models initialized")
	println("Now running $(n_pass) sddp passes")
	@showprogress for i in 1:n_pass
		ϵ_scenario = nim.ϵs[:,rand(1:S)]
		x₀ = [rand(x₀s)...]
		xscenario = forward_pass(m, ϵ_scenario, x₀, nim.fₜ)
		backward_pass!(m, V, xscenario)
	end
	return m
end