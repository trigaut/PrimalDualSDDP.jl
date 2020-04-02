struct PolyhedralFunction
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

	NonIslandedModel(Δt, capacity, ρc, ρd, 
					 pbmax, pbmin, pemax, 
					 Δhmax, cbuy, csell, 
					 α, β, ϵ, πϵ, ϵs)
end

function bellman_operator(nim::NonIslandedModel, t::Int, Vₜ₊₁::PolyhedralFunction)
	m = JuMP.direct_model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))
	
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
	@expression(m, net_dₜ₊₁, ϵ .+ nim.α[t] * net_dₜ + nim.β[t])

	importₜ₊₁ = net_dₜ₊₁ + uₜ+ .- uₜ⁻
	@variable(m, ebuy[1:nξ] >= 0.)
	@constraint(m, ebuy .>= importₜ₊₁)
	if nim.pemax < Inf
		@constraint(m, ebuy  .<= pemax)
	end
	esell = importₜ₊₁ .- ebuy

	@variable(m, θ[1:nξ])
	for (λ, γ) in eachcut(Vₜ₊₁)
		for ξ in 1:nξ
			@constraint(m, θ[ξ] >= λ'*[socₜ₊₁, hₜ₊₁, net_dₜ₊₁[ξ]] + γ)
		end
	end
	@objective(m, Min, sum(πϵ[t, ξ]* (nim.cbuy[t]*ebuy[ξ] + nim.csell[t]*esell[ξ] + θ[ξ]) for i in 1:nξ))

	return m
end

function solve!(m::JuMP.Model, socₜ::Float64, hₜ::Float64, net_dₜ::Float64)
	fix(m[:socₜ], socₜ)
	fix(m[:hₜ], hₜ)
	fix(m[:net_dₜ], net_dₜ)

	optimize!(m)
	
	@assert termination_status(m) == MOI.OPTIMAL

	return
end

function new_cut!(Vₜ::PolyhedralFunction, m::JuMP.Model, socₜ::Float64, hₜ::Float64, net_dₜ::Float64)
	solve!(m, socₜ, hₜ, net_dₜ)

	λ = [dual(m[:socₜ]), dual(m[:hₜ]), dual(m[:net_dₜ])]
	γ = objective_value(m) - λ'*[socₜ, hₜ, net_dₜ]

	cat!(V.λ, λ, dims = 1)
	push!(V.γ, γ)

	return
end

function update!(mₜ::JuMP.Model, Vₜ₊₁::PolyhedralFunction)
	nξ = length(mₜ[:θ])
	for ξ in 1:nξ
		@constraint(mₜ, mₜ[:θ][ξ] >= Vₜ₊₁.λ[end,:]'*[mₜ[:socₜ], mₜ[:hₜ], mₜ[:net_dₜ]] + Vₜ₊₁.γ[end])
	end
	return
end

function control!(m::JuMP.Model, socₜ::Float64, hₜ::Float64, net_dₜ::Float64)
	solve!(m, socₜ, hₜ, net_dₜ)

	♯uₜ⁺ = value(m[:uₜ⁺])
	♯uₜ⁻ = value(m[:uₜ⁻])

	return ♯uₜ⁺, ♯uₜ⁻
end

function fₜ!(m::JuMP.Model, socₜ::Float64, hₜ::Float64, net_dₜ::Float64)
	solve!(m, socₜ, hₜ, net_dₜ)

	♯socₜ₊₁ = value(m[:socₜ₊₁])
	♯hₜ₊₁ = value(m[:hₜ₊₁])
	♯net_dₜ₊₁ = value(m[:net_dₜ₊₁])

	return ♯socₜ₊₁, ♯hₜ₊₁, ♯net_dₜ₊₁
end

function forward_pass(m::Vector{JuMP.Model}, ϵscenario::Vector{Float64},
					  soc₀::Float64, h₀::Float64, net_d₀::Float64)
	T = length(ϵscenario)
	soc_scenario = fill(soc₀, T+1)
	h_scenario = fill(h₀, T+1)
	net_d_scenario = fill(net_d₀, T+1)
	for (t, ϵₜ₊₁) in enumerate(ϵscenario)
		socₜ₊₁, hₜ₊₁, net_dₜ₊₁ = fₜ!(m[t], socₜ, hₜ, net_dₜ)
	end
	return soc_scenario, h_scenario, net_d_scenario
end

function backward_pass!(m::Vector{JuMP.Model},
					  	V::Vector{PolyhedralFunction},
					  	soc_scenario::Vector{Float64}, 
					  	h₀::Vector{Float64}, 
					  	net_d₀::Vector{Float64})
	T = length(m)
	for t in T:-1:1
		socₜ = soc_scenario[t]
		hₜ = soc_scenario[t]
		net_dₜ = soc_scenario[t]
		new_cut!(V[t], m[t], socₜ, hₜ, net_dₜ)
	end
	return soc_scenario, h_scenario, net_d_scenario
end

function sddp!(nim::NonIslandedModel, n_pass::Int)
	S = size(nim.ϵs,2)
	for i in 1:n_pass
		ϵ_scenario = nim.ϵs[:,rand(1:S)] 
		soc_scenario, h_scenario, net_d_scenario = forward_pass(m, ϵ_scenario)
		backward_pass!(m, V, soc_scenario, h_scenario, net_d_scenario)
	end
	return m, V
end