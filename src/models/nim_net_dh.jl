# nim stands for Non Islanded Model

mutable struct NonIslandedModel <: DecisionHazardModel
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
	ξ
	πξ
	ξs
	fₜ
end

function NonIslandedModel(Δt::Float64, capacity::Float64, 
						  ρc::Float64, ρd::Float64, 
						  pbmax::Float64, pbmin::Float64, 
						  pemax::Float64, Δhmax::Float64, 
						  cbuy::Vector{Float64}, csell::Vector{Float64}, 
						  net_d_scenarios::Array{Float64,2}, bins::Int)
	
	α1, β1, _ = fitar_cholesky(net_d_scenarios)
	
	ξs1 = net_d_scenarios[2:end,:] .- α1 .* net_d_scenarios[1:end-1,:] .- β1

	α = cat([0.], α1, dims = 1)[:,1]
	β = cat([0.], β1, dims = 1)
	ξs = cat(net_d_scenarios[1:1,:], ξs1, dims = 1)

	ξ, πξ = discrete_white_noise(ξs, bins);

	function fₜ(t, xₜ, uₜ, ξₜ₊₁)
		xₜ₊₁ = [xₜ[1] + ρc*uₜ[1] - 1/ρd*uₜ[2],
				xₜ[2] - uₜ[1] - uₜ[2],
				α[t]*xₜ[3] + β[t] + ξₜ₊₁[1] ]
		return xₜ₊₁
	end

	NonIslandedModel(Δt, capacity, ρc, ρd, 
					 pbmax, pbmin, pemax, 
					 Δhmax, cbuy, csell, 
					 α, β, ξ, πξ, ξs[:,:,:], fₜ)
end

function bellman_operator(nim::NonIslandedModel, t::Int)

	m = JuMP.Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))
	ξ = nim.ξ[t]
	nξ = length(ξ)
	
	@variable(m, 0. <= uₜ⁺ <= nim.pbmax)
	@variable(m, 0. <= uₜ⁻ <= -nim.pbmin)

	@variable(m, 0. <= socₜ <= nim.capacity)
	@variable(m, socₜ₊₁)
	@constraint(m, socₜ₊₁ == socₜ + nim.ρc * uₜ⁺ - 1/nim.ρd * uₜ⁻)
	@constraint(m, socₜ + nim.ρc * uₜ⁺ - 1/nim.ρd * uₜ⁻ >= 0.)
	@constraint(m, socₜ + nim.ρc * uₜ⁺ - 1/nim.ρd * uₜ⁻ <= nim.capacity)

	@variable(m, 0 <= hₜ <= 2*nim.capacity)
	@variable(m, hₜ₊₁)
	@constraint(m, hₜ₊₁ == hₜ - uₜ⁺ - uₜ⁻)
	@constraint(m, hₜ₊₁ >= 0.)

	@variable(m, -1e2 <= net_dₜ <= 1e2)
	@variable(m, net_dₜ₊₁[i=1:nξ])
	@constraint(m, net_dₜ₊₁ .== ξ + nim.α[t] * net_dₜ + nim.β[t])

	@expression(m, importₜ₊₁[i=1:nξ], net_dₜ₊₁[i] + uₜ⁺ - uₜ⁻)
	@variable(m, ebuy[1:nξ] >= 0.)
	@constraint(m, ebuy .>= importₜ₊₁)
	if nim.pemax < Inf
		@constraint(m, ebuy  .<= nim.pemax)
	end
	@expression(m, esell[i=1:nξ], importₜ₊₁[i] - ebuy[i])

	@objective(m, Min, sum(nim.πξ[t][i]*(nim.cbuy[t]*ebuy[i] + nim.csell[t]*esell[i]) for i in 1:nξ))

	@expression(m, xₜ, [socₜ, hₜ, net_dₜ])
	@expression(m, xₜ₊₁[i=1:nξ], [socₜ₊₁, hₜ₊₁, net_dₜ₊₁[i]])
	@expression(m, uₜ, [uₜ⁺, uₜ⁻])

	return m
end

function dual_bellman_operator(nim::NonIslandedModel, t::Int)
	md = auto_dual_bellman_operator(nim, t, 162)
	set_optimizer(md, optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))

	md
end

# function intrapb(nim::NonIslandedModel, t::Int)
# 	LIP = 1e3

# 	ξ = nim.ξ[t]
# 	nξ = length(ξ)

# 	m = JuMP.Model()
	
# 	@variable(m, uₜ⁺)
# 	@variable(m, uₜ⁻)

# 	@variable(m, socₜ)
# 	@variable(m, socₜ₊₁)
# 	@variable(m, hₜ)
# 	@variable(m, hₜ₊₁)
# 	@variable(m, net_dₜ)
# 	@variable(m, net_dₜ₊₁[1:nξ])
# 	@variable(m, cut[1:nξ])
# 	@constraint(m, μS, socₜ₊₁ == socₜ + nim.ρc * uₜ⁺ - 1/nim.ρd * uₜ⁻)
# 	@constraint(m, μH, hₜ₊₁ == hₜ - uₜ⁺ - uₜ⁻)
# 	for i in 1:nξ
# 		con = @constraint(m, net_dₜ₊₁[i] == ξ[i] + nim.α[t] * net_dₜ + nim.β[t] + cut[i])
# 		JuMP.set_name(con, "μW[$(i)]")
# 	end

# 	@constraint(m, λ1, socₜ + nim.ρc * uₜ⁺ - 1/nim.ρd * uₜ⁻ >= 0.)
# 	@constraint(m, λ2, -socₜ - nim.ρc * uₜ⁺ + 1/nim.ρd * uₜ⁻ >= -nim.capacity)
# 	@constraint(m, λ3, hₜ - uₜ⁺ - uₜ⁻ >= 0.)

# 	@expression(m, importₜ₊₁[i=1:nξ], net_dₜ₊₁[i] + uₜ⁺ - uₜ⁻)
# 	@variable(m, ebuy[1:nξ])
# 	for i in 1:nξ
# 		con = @constraint(m, ebuy[i] >= 0)
# 		JuMP.set_name(con, "λ4[$(i)]")
# 		con = @constraint(m, ebuy[i] >= importₜ₊₁[i])
# 		JuMP.set_name(con, "λ5[$(i)]")
# 	end

# 	@constraint(m, λ6, uₜ⁺ >= 0.)
# 	@constraint(m, λ7, -uₜ⁺ >= -nim.pbmax)
# 	@constraint(m, λ8, uₜ⁻ >= 0.)
# 	@constraint(m, λ9, -uₜ⁻ >= nim.pbmin)
	
# 	for i in 1:nξ
# 		con = @constraint(m, cut[i] >= 0)
# 		JuMP.set_name(con, "λ10[$(i)]")
# 		con = @constraint(m, -(ξ[i] + nim.α[t] * net_dₜ + nim.β[t] + cut[i]) >= -LIP)
# 		JuMP.set_name(con, "λ11[$(i)]")
# 	end

# 	inst_cost = sum(nim.πξ[t][i]*(nim.cbuy[t]*ebuy[i]) for i in 1:nξ)

# 	@expression(m, xₜ, [socₜ, hₜ, net_dₜ])
# 	@expression(m, xₜ₊₁[i=1:nξ], [socₜ₊₁, hₜ₊₁, net_dₜ₊₁[i]])

# 	@objective(m, Min, sum(nim.πξ[t][i]*(nim.cbuy[t]*ebuy[i]) for i in 1:nξ) +
# 						sum(nim.πξ[t][i]*1e6*sum(xₜ₊₁[i]) for i in 1:nξ) +
# 						- 1e4*sum(xₜ))

# 	# JuMP.set_name.(FixRef.(xₜ),["μs","μh","μw"])
# 	@expression(m, uₜ, [uₜ⁺, uₜ⁻])

# 	return m
# end



# function dual_bellman_operator(nim::NonIslandedModel, t::Int)
# 	LIP = 1e10

# 	m = JuMP.Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))
	
# 	ξ = nim.ξ[t]
# 	nξ = length(ξ)

# 	ρc = nim.ρc
# 	ρd = nim.ρd
# 	πξ = nim.πξ[t]
# 	α = nim.α[t]
# 	cₜᵇ = nim.cbuy[t]
# 	S̄ = nim.capacity
# 	Ū = nim.pbmax
# 	Ulb = nim.pbmin

# 	@variable(m, λ1 >= 0)
# 	@variable(m, λ2 >= 0)
# 	@variable(m, λ3 >= 0)
# 	@variable(m, λ4[1:nξ] >= 0)
# 	@variable(m, λ5[1:nξ] >= 0)
# 	@variable(m, λ6 >= 0)
# 	@variable(m, λ7 >= 0)
# 	@variable(m, λ8 >= 0)
# 	@variable(m, λ9 >= 0)
# 	@variable(m, λ10[1:nξ] >= 0)
# 	@variable(m, λ11[1:nξ] >= 0)

# 	@variable(m, μs)
# 	@variable(m, μh)
# 	@variable(m, μw)

# 	@expression(m, μS, μs + λ1 - λ2)
# 	@expression(m, μH, μh + λ3)
# 	@variable(m, μW[1:nξ])
# 	@constraint(m, α*sum(πξ.*(μW .+ λ11)) == μw)
# 	@constraint(m, -μW .+ λ10 .- λ11 .== 0)

# 	@constraint(m, -ρc*μS + μH + ρc*(λ1 - λ2) - λ3 - sum(πξ.*λ5) + λ6 - λ7 == 0)
# 	@constraint(m, 1/ρd*μS + μH + 1/ρd*(λ2 - λ1) - λ3 + sum(πξ.*λ5) + λ8 - λ9 == 0)
	
# 	@constraint(m, λ4 .+ λ5 .== nim.cbuy[t])

# 	@variable(m, θ[1:nξ])
# 	@objective(m, Min, λ2*S̄ + λ7*Ū - λ9*Ulb + sum(πξ[i]*(100 .*λ11[i]-(ξ[i]+nim.β[t])*(μW[i]+λ11[i]) + θ[i]) for i in 1:nξ))

# 	@expression(m, μₜ, [μs, μh, μw])
# 	@expression(m, μₜ₊₁[i=1:nξ], [μS, μH, μW[i] - λ5[i]])
# 	@expression(m, νₜ, [λ1,λ2,λ3,λ4,λ5,λ6,λ7,λ8,λ9])

# 	@constraint(m, -LIP <= μS <= LIP)
# 	@constraint(m, -LIP <= μH <= LIP)
# 	for i in 1:nξ
# 		@constraint(m, μW[i] - λ5[i] <= LIP)
# 		@constraint(m, μW[i] - λ5[i] >= -LIP)
# 	end

# 	return m
# end