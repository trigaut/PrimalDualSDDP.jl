function simulation(hdm::HazardDecisionModel,
    m::Vector{JuMP.Model},
    ξscenarios::Array{Float64,3},
    x₀::Vector{Float64},
    lₜ::Function, 
    Kₜ::Function
    )
    T = size(ξscenarios, 1)
    n_pass = size(ξscenarios, 2)
    xscenarios = fill(0., T + 1, n_pass, length(x₀))
    costs = fill(0.,  n_pass)
    @inbounds for pass in 1:n_pass
        xₜ = x₀
        xscenarios[1, pass, :] .= x₀
        for (t, ξₜ₊₁) in enumerate(eachrow(ξscenarios[:, pass, :]))
            uₜ₊₁ = control!(hdm, m[t], xₜ, collect(ξₜ₊₁))
            costs[pass] += lₜ(t, xₜ, uₜ₊₁, ξₜ₊₁)
            xₜ = hdm.fₜ(t, xₜ, uₜ₊₁, ξₜ₊₁)
            xscenarios[t + 1, pass, :] .= xₜ
        end
        costs[pass] += Kₜ(xₜ)
    end
    return xscenarios, costs
end