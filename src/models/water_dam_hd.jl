mutable struct WaterDamModel <: HazardDecisionModel
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
    
    ξ, πξ = discrete_white_noise(rainfall_scenarios, bins)

    function fₜ(t, xₜ, uₜ₊₁, ξₜ₊₁)
        return [xₜ[1] - uₜ₊₁[1] + ξₜ₊₁[1]]
    end

    return WaterDamModel(Δt, capacity, umax, csell, 
                         ξ, πξ, rainfall_scenarios[:,:,:], fₜ)
end

function bellman_operator(wdm::WaterDamModel, t::Int)
    m = Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))

    @variable(m, 0 <= lₜ <= wdm.capacity)
    @variable(m, 0 <= lₜ₊₁ <= wdm.capacity)
    @variable(m, rₜ₊₁)
    @variable(m, 0 <= u <= wdm.umax)
    @variable(m, s >= 0)
    @constraint(m, lₜ₊₁ == lₜ - u .+ rₜ₊₁ - s)
    @objective(m, Min, -wdm.csell[t]*u)

    @expression(m, xₜ, [lₜ])
    @expression(m, uₜ₊₁, [u, s])
    @expression(m, xₜ₊₁, [lₜ₊₁])
    @expression(m, ξₜ₊₁, [rₜ₊₁])

    return m
end

function dual_bellman_operator(wdm::WaterDamModel, t::Int)
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
    
    md = auto_dual_bellman_operator(m, 10.)
    set_optimizer(md, optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))

    return md
end
