# nim stands for Non Islanded Model
const LIP = 162.

mutable struct NonIslandedModel2 <: DecisionHazardModel
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
	ξ
	πξ
	ξs
	fₜ
end

function NonIslandedModel2(Δt::Float64, capacity::Float64, 
						  ρc::Float64, ρd::Float64, 
						  pbmax::Float64, pbmin::Float64, 
						  pemax::Float64, Δhmax::Float64, 
						  cbuy::Vector{Float64}, csell::Vector{Float64}, 
						  scenarios::Array{Float64,3}, bins::Int)
	
	αd = fitar_ridge_reg(scenarios[:,:,1])
	αp = fitar_ridge_reg(scenarios[:,:,2])
	
	ξsd = scenarios[2:end,:,1] .- αd .* scenarios[1:end-1,:,1]
	ξsp = scenarios[2:end,:,2] .- αp .* scenarios[1:end-1,:,2]

	αd = cat([0.], αd, dims = 1)[:,1]
	ξsd = cat(scenarios[1:1,:,1], ξsd, dims = 1)

	αp = cat([0.], αp, dims = 1)[:,1]
	ξsp = cat(scenarios[1:1,:,2], ξsp, dims = 1)
	
	α = cat(αd[:,:], αp[:,:], dims =2)
	ξs = cat(ξsd[:,:,:], ξsp[:,:,:], dims = 3)

	ξ, πξ = discrete_white_noise(ξs, bins);

	function fₜ(t, xₜ, uₜ, ξₜ₊₁)
		xₜ₊₁ = [xₜ[1] + ρc*uₜ[1] - 1/ρd*uₜ[2],
				xₜ[2] - uₜ[1] - uₜ[2],
				αd[t]*xₜ[3] + ξₜ₊₁[1],
				αp[t]*xₜ[4] + ξₜ₊₁[2] ]
		return xₜ₊₁
	end

	NonIslandedModel2(Δt, capacity, ρc, ρd, 
					 pbmax, pbmin, pemax, 
					 Δhmax, cbuy, csell, 
					 α, ξ, πξ, ξs, fₜ)
end

function bellman_operator(nim::NonIslandedModel2, t::Int)

	m = JuMP.Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))
	
	@variable(m, 0. <= uₜ⁺ <= nim.pbmax)
	@variable(m, 0. <= uₜ⁻ <= -nim.pbmin)

	@variable(m, 0 <= socₜ <= nim.capacity)
	@variable(m, 0 <= socₜ₊₁ <= nim.capacity)
	@constraint(m, socₜ₊₁ == socₜ + nim.ρc * uₜ⁺ - 1/nim.ρd * uₜ⁻)

	@variable(m, 0 <= hₜ <= 1e3)
	@variable(m, 0 <= hₜ₊₁ <= 1e3)
	@constraint(m, hₜ₊₁ == hₜ - uₜ⁺ - uₜ⁻)

	ξ = nim.ξ[t]
	nξ = size(ξ,1)
	@variable(m, 0 <= dₜ <= 1e2)
	@variable(m, 0 <= pₜ <= 1e2)
	@variable(m, dₜ₊₁[i=1:nξ] )
	@variable(m, pₜ₊₁[i=1:nξ] )
	
	@constraint(m, dₜ₊₁ .== ξ[:,1] .+ nim.α[t,1] * dₜ)
	@constraint(m, pₜ₊₁ .== ξ[:,2] .+ nim.α[t,2] * pₜ)

	@expression(m, importₜ₊₁[i=1:nξ], dₜ₊₁[i] - pₜ₊₁[i] + uₜ⁺ - uₜ⁻)
	@variable(m, ebuy[1:nξ] >= 0.)
	@constraint(m, ebuy .>= importₜ₊₁)
	if nim.pemax < Inf
		@constraint(m, ebuy  .<= nim.pemax)
	end
	@expression(m, esell[i=1:nξ], importₜ₊₁[i] - ebuy[i])
	@expression(m, xₜ, [socₜ, hₜ, dₜ, pₜ])

	@objective(m, Min, sum(nim.πξ[t][i]*(nim.cbuy[t]*ebuy[i] + nim.csell[t]*esell[i]) for i in 1:nξ))

	@expression(m, xₜ₊₁[i=1:nξ], [socₜ₊₁, hₜ₊₁, dₜ₊₁[i], pₜ₊₁[i]])
	@expression(m, uₜ, [uₜ⁺, uₜ⁻])

	return m
end

function dual_bellman_operator(nim::NonIslandedModel2, t::Int)
	m = bellman_operator(nim, t)
	md = dualize(m; dual_names = DualNames("", ""))

	nx = length(m[:xₜ]) 
	info =  VariableInfo(false, NaN, false, NaN, false, NaN, 
						 false, NaN, false, false)

	md[:μₜ] = VariableRef[]
	for var in name.(m[:xₜ])
		con = JuMP.constraint_by_name(md, var)
		adjointvar = add_variable(md, build_variable(error, info), "μ"*var)
		push!(md[:μₜ], adjointvar)
		set_normalized_coefficient(con, adjointvar, 1.)
	end

	xₜ₊₁uniquenames = unique(cat(map(t -> name.(t), m[:xₜ₊₁])..., dims=1))

	info =  VariableInfo(true, -LIP, true, LIP, false, NaN, 
						 false, NaN, false, false)

	for var in xₜ₊₁uniquenames
		con = JuMP.constraint_by_name(md, var)
		adjointvar = add_variable(md, build_variable(error, info), "μ"*var)
		set_normalized_coefficient(con, adjointvar, -1.)
	end
	
	md[:μₜ₊₁] = Array{Array{VariableRef,1},1}()
	for	var_array in m[:xₜ₊₁]
		push!(md[:μₜ₊₁], variable_by_name.(md, "μ".*name.(var_array)))
	end

	obj_expr = objective_function(md)
	@objective(md, Min, -obj_expr)

	set_optimizer(md, optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))

	md
end