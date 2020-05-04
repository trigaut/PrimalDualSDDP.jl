using JuMP, PrimalDualSDDP, Clp

mutable struct WaterDamModel <: PrimalDualSDDP.HazardDecisionModel
    Δt::Float64
    capacity::Float64
    umax::Float64
    csell::Vector{Float64}
    ξ
    πξ
    ξs
    fₜ
end

function WaterDamModel(Δt::Float64, capacity::Float64, 
                       umax::Float64, csell::Vector{Float64}, 
                       rainfall_scenarios::Array{Float64,2}, 
                       bins::Int)
    
    ξ, πξ = PrimalDualSDDP.discrete_white_noise(rainfall_scenarios, bins)

    function fₜ(t, xₜ, uₜ₊₁, ξₜ₊₁)
        return [xₜ[1] - uₜ₊₁[1] + ξₜ₊₁[1] - uₜ₊₁[2]]
    end

    return WaterDamModel(Δt, capacity, umax, csell, 
                         ξ, πξ, rainfall_scenarios[:,:,:], fₜ)
end

function bellman_operator(wdm::WaterDamModel, t::Int)
    m = Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))

    @variable(m, 0 <= l1 <= wdm.capacity)
    @variable(m, 0 <= l2 <= wdm.capacity)
    @variable(m, r2)
    @variable(m, 0 <= u <= wdm.umax)
    @variable(m, s >= 0)
    @constraint(m, l2 == l1 - u .+ r2 - s)
    @objective(m, Min, -wdm.csell[t]*u)

    @expression(m, xₜ, [l1])
    @expression(m, uₜ₊₁, [u, s])
    @expression(m, xₜ₊₁, [l2])
    @expression(m, ξₜ₊₁, [r2])

    return m
end

function dual_bellman_operator(wdm::WaterDamModel, t::Int, 
                              l1_regularization::Real)
    m = Model()

    nξ = size(wdm.ξ[t],1)

    @variable(m, 0 <= lₜ <= wdm.capacity)
    @variable(m, 0 <= lₜ₊₁[1:nξ] <= wdm.capacity)
    
    @variable(m, 0 <= u[1:nξ] <= wdm.umax)
    @variable(m, s[1:nξ] >= 0)
    @constraint(m, lₜ₊₁ .== wdm.ξ[t][:,1] .- u .+ lₜ .- s)
    @objective(m, Min, -wdm.csell[t] * wdm.πξ[t]'*u)

    @expression(m, xₜ, [lₜ])
    @expression(m, xₜ₊₁[i=1:nξ], [lₜ₊₁[i]])
    
    md = PrimalDualSDDP.auto_dual_bellman_operator(m, wdm.πξ[t], l1_regularization)
    set_optimizer(md, optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))

    return md
end
