abstract type DecisionHazardModel <: LinearBellmanModel end

function initialize_lift_primal!(m::JuMP.Model,
                                 dhm::DecisionHazardModel,
                                 t::Int,
                                 Vₜ₊₁::PolyhedralFunction)
    nξ = length(m[:xₜ₊₁])
    @variable(m, θ[1:nξ])
    for (λ, γ) in eachcut(Vₜ₊₁)
        for i in 1:nξ
            @constraint(m, θ[i] >= λ'*m[:xₜ₊₁][i] + γ)
        end
    end
    obj_expr = objective_function(m)
    @objective(m, Min, obj_expr + sum(dhm.πξ[t][i]*θ[i] for i in 1:nξ))
    return
end

function primalsolve!(::DecisionHazardModel,
                       m::JuMP.Model,
                       xₜ::Vector{Float64})
    fix.(m[:xₜ], xₜ, force=true)
    optimize!(m)
    @assert termination_status(m) == MOI.OPTIMAL println(m)
    return
end

function new_cut!(dhm::DecisionHazardModel,
                  Vₜ::PolyhedralFunction,
                  m::JuMP.Model,
                  xₜ::Vector{Float64})
    primalsolve!(dhm, m, xₜ)
    λ = dual.(FixRef.(m[:xₜ]))
    γ = objective_value(m) - λ'*xₜ
    Vₜ.λ = cat(Vₜ.λ, λ', dims = 1)
    push!(Vₜ.γ, γ)
    return
end

function update!(dhm::DecisionHazardModel,
                 mₜ::JuMP.Model,
                 Vₜ₊₁::PolyhedralFunction)
    nξ = length(mₜ[:θ])
    for i in 1:nξ
        @constraint(mₜ, mₜ[:θ][i] >= Vₜ₊₁.λ[end,:]'*mₜ[:xₜ₊₁][i] + Vₜ₊₁.γ[end])
    end
    return
end

function control!(dhm::DecisionHazardModel,
                  m::JuMP.Model,
                  xₜ::Vector{Float64})
    primalsolve!(dhm, m, xₜ)
    return value.(m[:uₜ])
end

function forward_pass(dhm::DecisionHazardModel,
                      m::Vector{JuMP.Model},
                      ξscenarios::Array{Float64, 3},
                      x₀::Vector{Float64})
    T = size(ξscenarios,1)
    n_pass = size(ξscenarios,2)
    xscenarios = fill(0., T+1, n_pass, length(x₀))
    @inbounds for pass in 1:n_pass
        xₜ = x₀
        xscenarios[1,pass,:] .= x₀
        for (t, ξₜ₊₁) in enumerate(eachrow(ξscenarios[:,pass,:]))
            uₜ = control!(dhm, m[t], xₜ)
            xₜ = dhm.fₜ(t, xₜ, uₜ, ξₜ₊₁)
            xscenarios[t+1,pass,:] .= xₜ
        end
    end
    return xscenarios
end

function backward_pass!(dhm::DecisionHazardModel,
                        m::Vector{JuMP.Model},
                        V::Vector{PolyhedralFunction},
                        xscenarios::Array{Float64,3})
    T = length(m)
    n_pass = size(xscenarios, 2)
    for pass in 1:n_pass
        new_cut!(dhm, V[T], m[T], xscenarios[T,pass,:])
    end
    @inbounds for t in T-1:-1:1
        update!(dhm, m[t], V[t+1])
        for pass in 1:n_pass
            new_cut!(dhm, V[t], m[t], xscenarios[t,pass,:])
        end
    end
    return
end

function primalsddp!(dhm::DecisionHazardModel,
                     V::Array{PolyhedralFunction},
                     n_pass::Int,
                     x₀s::Array;
                     nprune::Int=n_pass,
                     pruner=nothing,
                     verbose::Int=n_pass)

    n_pruning = div(n_pass, nprune)
    println("** Primal SDDP with $(n_pass) passes, $(n_pruning) pruning  **")
    if n_pruning > 0 && isnothing(pruner)
        error("Could not proceed to pruning as `pruner` is not set.")
    end

    T, S = size(dhm.ξs)
    m = [bellman_operator(dhm, t) for t in 1:T]
    for (t, Vₜ₊₁) in enumerate(V[2:end])
        initialize_lift_primal!(m[t], dhm, t, Vₜ₊₁)
    end

    println("Primal Bellman JuMP Models initialized")
    println("Now running SDDP passes")

    @showprogress for i in 1:n_pass
        ξscenarios = dhm.ξs[:,rand(1:S,1),:]
        x₀ = [rand(x₀s)...]
        xscenarios = forward_pass(dhm, m, ξscenarios, x₀)
        backward_pass!(dhm, m, V, xscenarios)

        if mod(i,nprune) == 0 && !isnothing(pruner)
            println("\n Performing pruning number $(div(i,nprune))")
            for (t, Vₜ₊₁) in enumerate(V[2:end])
                prune!(V[t+1], pruner)
                m[t] = bellman_operator(dhm, t)
                initialize_lift_primal!(m[t], dhm, t, V[t+1])
            end
        end
        if mod(i, verbose) == 0
            lb = V[1](x₀)
            println("Iter $i    lb ", lb)
        end
    end
    if !isnothing(pruner)
        prune!(V[1], pruner)
    end
    return m
end
