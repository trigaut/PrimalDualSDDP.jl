abstract type HazardDecisionModel <: LinearBellmanModel end

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

function new_cut!(hdm::HazardDecisionModel,
                  Vₜ::PolyhedralFunction,
                  m::JuMP.Model,
                  t::Int,
                  xₜ::Vector{Float64})
    ξₜ₊₁ = hdm.ξ[t]
    πₜ₊₁ = hdm.πξ[t]
    λ = zeros(length(xₜ))
    γ = 0.
    for (i, πᵢ) in enumerate(πₜ₊₁)
        primalsolve!(hdm, m, xₜ, ξₜ₊₁[i,:])
        λ .= λ .+ πᵢ.*dual.(FixRef.(m[:xₜ]))
        γ += πᵢ.*(objective_value(m) - λ'*xₜ)
    end
    Vₜ.λ = cat(Vₜ.λ, λ', dims = 1)
    push!(Vₜ.γ, γ)
    return λ
end

function update!(hdm::HazardDecisionModel,
                 mₜ::JuMP.Model,
                 Vₜ₊₁::PolyhedralFunction)
    @constraint(mₜ, mₜ[:θ] >= Vₜ₊₁.λ[end,:]'*mₜ[:xₜ₊₁] + Vₜ₊₁.γ[end])
    return
end

function control!(hdm::HazardDecisionModel,
                  m::JuMP.Model,
                  xₜ::Vector{Float64},
                  ξₜ₊₁::Vector{Float64})
    primalsolve!(hdm, m, xₜ, ξₜ₊₁)
    return value.(m[:uₜ₊₁])
end

function forward_pass(hdm::HazardDecisionModel,
                      m::Vector{JuMP.Model},
                      ξscenarios::Array{Float64, 3},
                      x₀::Vector{Float64})
    T = size(ξscenarios,1)
    n_pass = size(ξscenarios,2)
    xscenarios = fill(0., T+1, n_pass, length(x₀))
    @inbounds for pass in 1:n_pass
        xₜ = x₀
        xscenarios[1, pass, :] .= x₀
        for (t, ξₜ₊₁) in enumerate(eachrow(ξscenarios[:, pass, :]))
            uₜ₊₁ = control!(hdm, m[t], xₜ, collect(ξₜ₊₁))
            xₜ = hdm.fₜ(t, xₜ, uₜ₊₁, ξₜ₊₁)
            xscenarios[t+1, pass, :] .= xₜ
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
    costates = zeros(size(xscenarios))
    for pass in 1:n_pass
        λ = new_cut!(hdm, V[T], m[T], T, xscenarios[T, pass,:])
        costates[T, pass, :] .= λ
    end
    @inbounds for t in T-1:-1:1
        update!(hdm, m[t], V[t+1])
        for pass in 1:n_pass
            λ = new_cut!(hdm, V[t], m[t], t, xscenarios[t, pass,:])
            costates[t, pass, :] .= λ
        end
    end
    return costates
end

function primalsddp!(hdm::HazardDecisionModel,
                     V::Array{PolyhedralFunction},
                     n_pass::Int,
                     x₀s::Array;
                     nprune::Int = n_pass,
                     pruner=nothing,
                     verbose::Int=n_pass)
    n_pruning = div(n_pass, nprune)
    println("** Primal SDDP, in Hazard Decision , with $(n_pass) passes and $(n_pruning) pruning  **")
    if n_pruning > 0 && isnothing(pruner)
        error("Could not proceed to pruning as `pruner` is not set.")
    end

    T, S = size(hdm.ξs)
    m = [bellman_operator(hdm, t) for t in 1:T]
    for (t, Vₜ₊₁) in enumerate(V[2:end])
        initialize_lift_primal!(m[t], hdm, t, Vₜ₊₁)
    end

    println("Primal Bellman JuMP Models initialized")
    println("Now running SDDP passes")

    @showprogress for i in 1:n_pass
        ξscenarios = hdm.ξs[:, rand(1:S,1), :]
        x₀ = [rand(x₀s)...]
        xscenarios = forward_pass(hdm, m, ξscenarios, x₀)
        backward_pass!(hdm, m, V, xscenarios)

        if mod(i, nprune) == 0
            println("\n Performing pruning number $(div(i, nprune))")
            for (t, Vₜ₊₁) in enumerate(V[2:end])
                V[t+1] = unique(Vₜ₊₁)
                exact_pruning!(V[t+1], pruner)
                m[t] = bellman_operator(hdm, t)
                initialize_lift_primal!(m[t], hdm, t, V[t+1])
            end
        end
        if mod(i, verbose) == 0
            lb = V[1](x₀)
            println("Iter $i    lb ", lb)
        end
    end
    prune!(V[1], pruner)
    return m
end
