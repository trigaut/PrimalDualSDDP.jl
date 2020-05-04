# nim stands for Non Islanded Model
using JuMP, PrimalDualSDDP, Clp

mutable struct NonIslandedNetHDModel <: PrimalDualSDDP.HazardDecisionModel
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
    ξ
    πξ
    ξs
    fₜ
end

function NonIslandedNetHDModel(Δt::Float64, capacity::Float64, 
                          ρc::Float64, ρd::Float64, 
                          pbmax::Float64, pbmin::Float64, 
                          pemax::Float64, Δhmax::Float64, 
                          cbuy::Vector{Float64}, csell::Vector{Float64}, 
                          net_d_scenarios::Array{Float64,2}, bins::Int)
    ξs = net_d_scenarios    
    ξ, πξ = PrimalDualSDDP.discrete_white_noise(ξs, bins);

    function fₜ(t, xₜ, uₜ, ξₜ₊₁)
        xₜ₊₁ = [xₜ[1] + ρc*uₜ[1] - 1/ρd*uₜ[2],
                xₜ[2] - uₜ[1] - uₜ[2]]
        return xₜ₊₁
    end

    NonIslandedNetHDModel(Δt, capacity, ρc, ρd, 
                         pbmax, pbmin, pemax, 
                         Δhmax, cbuy, csell, 
                         ξ, πξ, ξs[:,:,:], fₜ)
end

function bellman_operator(nim::NonIslandedNetHDModel, t::Int)

    m = JuMP.Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))
    ξ = nim.ξ[t]
    nξ = length(ξ)
    
    @variable(m, 0. <= uₜ⁺ <= nim.pbmax)
    @variable(m, 0. <= uₜ⁻ <= -nim.pbmin)

    @variable(m, 0. <= socₜ <= nim.capacity)
    @variable(m, 0. <= socₜ₊₁ <= nim.capacity)
    @constraint(m, socₜ₊₁ == socₜ + nim.ρc * uₜ⁺ - 1/nim.ρd * uₜ⁻)

    @variable(m, 0 <= hₜ <= 2*nim.capacity)
    @variable(m, hₜ₊₁ >= 0)
    @constraint(m, hₜ₊₁ == hₜ - uₜ⁺ - uₜ⁻)
    @variable(m, net_dₜ₊₁)

    @expression(m, importₜ₊₁, net_dₜ₊₁ + uₜ⁺ - uₜ⁻)
    @variable(m, ebuy >= 0.)
    @constraint(m, ebuy .>= importₜ₊₁)
    if nim.pemax < Inf
        @constraint(m, ebuy  .<= nim.pemax)
    end
    @expression(m, esell, importₜ₊₁ - ebuy)

    @objective(m, Min, nim.cbuy[t]*ebuy + nim.csell[t]*esell)

    @expression(m, xₜ, [socₜ, hₜ])
    @expression(m, xₜ₊₁, [socₜ₊₁, hₜ₊₁])
    @expression(m, uₜ₊₁, [uₜ⁺, uₜ⁻])
    @expression(m, ξₜ₊₁, [net_dₜ₊₁])

    return m
end

function dual_bellman_operator(nim::NonIslandedNetHDModel, 
                               t::Int, 
                               l1_regularization::Real)

    m = JuMP.Model()
    ξ = nim.ξ[t]
    nξ = length(ξ)
    
    @variable(m, 0. <= uₜ⁺[1:nξ] <= nim.pbmax)
    @variable(m, 0. <= uₜ⁻[1:nξ] <= -nim.pbmin)

    @variable(m, 0. <= socₜ <= nim.capacity)
    @variable(m, 0. <= socₜ₊₁[1:nξ] <= nim.capacity)
    @constraint(m, socₜ₊₁ .== nim.ρc .* uₜ⁺ .- 1/nim.ρd .* uₜ⁻ .+ socₜ)

    @variable(m, 0 <= hₜ <= 2*nim.capacity)
    @variable(m, hₜ₊₁[1:nξ] >= 0.)
    @constraint(m, hₜ₊₁ .== - uₜ⁺ - uₜ⁻ + hₜ)

    @expression(m, importₜ₊₁[i=1:nξ], ξ[i] + uₜ⁺[i] - uₜ⁻[i])
    @variable(m, ebuy[1:nξ] >= 0.)
    @constraint(m, ebuy .>= importₜ₊₁)
    if nim.pemax < Inf
        @constraint(m, ebuy  .<= nim.pemax)
    end
    @expression(m, esell[i=1:nξ], importₜ₊₁[i] - ebuy[i])

    @objective(m, Min, sum(nim.πξ[t][i]*(nim.cbuy[t]*ebuy[i] + nim.csell[t]*esell[i]) for i in 1:nξ))

    @expression(m, xₜ, [socₜ, hₜ])
    @expression(m, xₜ₊₁[i=1:nξ], [socₜ₊₁[i], hₜ₊₁[i]])

    md = PrimalDualSDDP.auto_dual_bellman_operator(m, nim.πξ[t], l1_regularization)
    set_optimizer(md, optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))

    md
end
