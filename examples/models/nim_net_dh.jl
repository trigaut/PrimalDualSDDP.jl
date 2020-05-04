# nim stands for Non Islanded Model
using JuMP, PrimalDualSDDP, Clp

mutable struct NonIslandedNetModel <: PrimalDualSDDP.DecisionHazardModel
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

function NonIslandedNetModel(Δt::Float64, capacity::Float64, 
                          ρc::Float64, ρd::Float64, 
                          pbmax::Float64, pbmin::Float64, 
                          pemax::Float64, Δhmax::Float64, 
                          cbuy::Vector{Float64}, csell::Vector{Float64}, 
                          net_d_scenarios::Array{Float64,2}, bins::Int)
    
    α1, β1, _ = PrimalDualSDDP.fitar_cholesky(net_d_scenarios)
    
    ξs1 = net_d_scenarios[2:end,:] .- α1 .* net_d_scenarios[1:end-1,:] .- β1

    α = cat([0.], α1, dims = 1)[:,1]
    β = cat([0.], β1, dims = 1)
    ξs = cat(net_d_scenarios[1:1,:], ξs1, dims = 1)

    ξ, πξ = PrimalDualSDDP.discrete_white_noise(ξs, bins);

    function fₜ(t, xₜ, uₜ, ξₜ₊₁)
        xₜ₊₁ = [xₜ[1] + ρc*uₜ[1] - 1/ρd*uₜ[2],
                xₜ[2] - uₜ[1] - uₜ[2],
                α[t]*xₜ[3] + β[t] + ξₜ₊₁[1] ]
        return xₜ₊₁
    end

    NonIslandedNetModel(Δt, capacity, ρc, ρd, 
                     pbmax, pbmin, pemax, 
                     Δhmax, cbuy, csell, 
                     α, β, ξ, πξ, ξs[:,:,:], fₜ)
end

function bellman_operator(nim::NonIslandedNetModel, t::Int)

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

function dual_bellman_operator(nim::NonIslandedNetModel, 
                               t::Int,
                               l1_regularization::Real)
    md = PrimalDualSDDP.auto_dual_bellman_operator(nim, t, l1_regularization)
    set_optimizer(md, optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))

    md
end
