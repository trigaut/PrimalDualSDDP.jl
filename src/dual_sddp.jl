abstract type AbstractInitialSampler end

sample(::AbstractInitialSampler) = nothing

struct FixedInitSampler <: AbstractInitialSampler
    init_positions::Array
    n_positions::Int
end
FixedInitSampler(xs::Array) = FixedInitSampler(xs, length(xs))

function sample(s::FixedInitSampler)
    if s.n_positions == 1
        return s.init_positions[1]
    else
        return [rand(s.init_positions)...]
    end
end

function initialize_lift_dual!(m::JuMP.Model,
                               lbm::LinearBellmanModel,
                               t::Int,
                               Dₜ₊₁::PolyhedralFunction)
    nξ = length(m[:μₜ₊₁])
    @variable(m, θ[1:nξ])
    for (λ, γ) in eachcut(Dₜ₊₁)
        for i in 1:nξ
            @constraint(m, θ[i] >= λ'*m[:μₜ₊₁][i] + γ)
        end
    end
    obj_expr = objective_function(m)
    @objective(m, Min, obj_expr + sum(lbm.πξ[t][i]*θ[i] for i in 1:nξ))
    return
end

function dualsolve!(::LinearBellmanModel,
                    m::JuMP.Model,
                    μₜ::Vector{Float64})
    fix.(m[:μₜ], μₜ, force=true)
    optimize!(m)
    @assert termination_status(m) == MOI.OPTIMAL println(m)
    return
end

function dual_new_cut!(lbm::LinearBellmanModel,
                       Dₜ::PolyhedralFunction,
                       m::JuMP.Model,
                       μₜ::Vector{Float64})
    dualsolve!(lbm, m, μₜ)
    λ = dual.(FixRef.(m[:μₜ]))
    γ = objective_value(m) - λ'*μₜ
    Dₜ.λ = cat(Dₜ.λ, λ', dims = 1)
    push!(Dₜ.γ, γ)
    return
end

function dualupdate!(lbm::LinearBellmanModel,
                     mₜ::JuMP.Model,
                     Dₜ₊₁::PolyhedralFunction)
    nξ = length(mₜ[:θ])
    for i in 1:nξ
        @constraint(mₜ, mₜ[:θ][i] >= Dₜ₊₁.λ[end,:]'*mₜ[:μₜ₊₁][i] + Dₜ₊₁.γ[end])
    end
    return
end

function dualstate!(lbm::LinearBellmanModel,
                    m::JuMP.Model,
                    μₜ::Vector{Float64})
    dualsolve!(lbm, m, μₜ)
    return map(μ♯ -> value.(μ♯), m[:μₜ₊₁])
end

function dualstatecuts!(lbm::LinearBellmanModel,
                    Dₜ::PolyhedralFunction,
                    m::JuMP.Model,
                    μₜ::Vector{Float64})
    dualsolve!(lbm, m, μₜ)
    λ = dual.(FixRef.(m[:μₜ]))
    γ = objective_value(m) - λ'*μₜ
    Dₜ.λ = cat(Dₜ.λ, λ', dims = 1)
    push!(Dₜ.γ, γ)
    return map(μ♯ -> value.(μ♯), m[:μₜ₊₁])
end

function dual_forward_pass(lbm::LinearBellmanModel,
                           m::Vector{JuMP.Model},
                           ξscenarios::Array{Float64, 3},
                           μ₀::Vector{Float64})
    T = size(ξscenarios,1)
    n_pass = size(ξscenarios,2)
    μscenarios = fill(0., T+1, n_pass, length(μ₀))
    @inbounds for pass in 1:n_pass
        μₜ = μ₀
        μscenarios[1, pass, :] .= μ₀
        for (t, ξₜ₊₁) in enumerate(eachrow(ξscenarios[:, pass, :]))
            μₜ₊₁ = dualstate!(lbm, m[t], μₜ)
            μscenarios[t+1, pass,:] .= rand(μₜ₊₁)
            μₜ = μscenarios[t+1, pass, :]
        end
    end
    return μscenarios
end

function dual_cupps_pass(lbm::LinearBellmanModel,
                         m::Vector{JuMP.Model},
                         ξscenarios::Array{Float64, 3},
                         D::Vector{PolyhedralFunction},
                         μ₀::Vector{Float64})
    T = size(ξscenarios,1)
    n_pass = size(ξscenarios,2)
    μscenarios = fill(0., T+1, n_pass, length(μ₀))
    @inbounds for pass in 1:n_pass
        μₜ = μ₀
        μscenarios[1, pass, :] .= μ₀
        for (t, ξₜ₊₁) in enumerate(eachrow(ξscenarios[:, pass, :]))
            μₜ₊₁ = dualstatecuts!(lbm, D[t], m[t], μₜ)
            if t > 1
                dualupdate!(lbm, m[t-1], D[t])
            end
            μscenarios[t+1, pass,:] .= rand(μₜ₊₁)
            μₜ = μscenarios[t+1, pass, :]
        end
    end
    return μscenarios
end

function dual_backward_pass!(lbm::LinearBellmanModel,
                             m::Vector{JuMP.Model},
                             D::Vector{PolyhedralFunction},
                             μscenarios::Array{Float64,3})
    T = length(m)
    n_pass = size(μscenarios, 2)
    for pass in 1:n_pass
        dual_new_cut!(lbm, D[T], m[T], μscenarios[T,pass,:])
    end
    @inbounds for t in T-1:-1:1
        dualupdate!(lbm, m[t], D[t+1])
        for pass in 1:n_pass
            dual_new_cut!(lbm, D[t], m[t], μscenarios[t,pass,:])
        end
    end
    return
end

# TODO: currently we rebuild a JuMP Model each time we recompute the
# initial position. We should find a better way to keep them in sync
# with new cuts added.
function initial_position(D, seed, optimizer_constructor, l1_regularization)
    model = JuMP.Model(optimizer_constructor)
    JuMP.set_silent(model)
    x₀ = sample(seed)
    fenchel_transform_as_sup(model, D, x₀, l1_regularization)
    JuMP.optimize!(model)
    return JuMP.value.(model[:λ])
end

function dualsddp!(lbm::LinearBellmanModel,
                   D::Array{PolyhedralFunction},
                   n_pass::Int,
                   seed::AbstractInitialSampler;
                   nprune::Int = n_pass,
                   pruner=nothing,
                   prunetol::Real = 0.,
                   verbose::Int=n_pass+1,
                   l1_regularization= 1e6)
    n_pruning = div(n_pass, nprune)

    println("** Dual SDDP with $(n_pass) passes, $(n_pruning) pruning  **")
    if n_pruning > 0 && isnothing(pruner)
        error("Could not proceed to pruning as `pruner` is not set.")
    end

    T, S = size(lbm.ξs)
    if isa(l1_regularization, Vector)
        m = [dual_bellman_operator(lbm, t, l1_regularization[t]) for t in 1:T]
    else
        m = [dual_bellman_operator(lbm, t, l1_regularization) for t in 1:T]
    end

    for (t, Dₜ₊₁) in enumerate(D[2:end])
        initialize_lift_dual!(m[t], lbm, t, Dₜ₊₁)
    end

    println("Dual Bellman JuMP Models initialized")
    println("Now running SDDP passes")

    for i in 1:n_pass
        ξscenarios = lbm.ξs[:, rand(1:S,1), :]
        l1_reg = isa(l1_regularization, Vector) ? l1_regularization[1] : l1_regularization
        μ₀ = initial_position(D[1], seed, pruner.optimizer_constructor, l1_reg)
        μscenarios = dual_forward_pass(lbm, m, ξscenarios, μ₀)
        dual_backward_pass!(lbm, m, D, μscenarios)

        if mod(i, nprune) == 0
            println("\n Performing pruning number $(div(i, nprune))")
            for (t, Dₜ₊₁) in enumerate(D[2:end])
                prune!(D[t+1], pruner)
                l1_reg = isa(l1_regularization, Vector) ? l1_regularization[t] : l1_regularization
                m[t] = dual_bellman_operator(lbm, t, l1_reg)
                initialize_lift_dual!(m[t], lbm, t, D[t+1])
            end
        end
        if mod(i, verbose) == 0
            l1_reg = isa(l1_regularization, Vector) ? l1_regularization[1] : l1_regularization
            v₀ = PolyhedralFenchelTransform(D[1], l1_reg)
            # TODO
            lb = v₀(μ₀, pruner.optimizer_constructor)
            println("Iter $i    lb ", lb)
        end
    end
    return m
end

function primaldualsddp!(model::LinearBellmanModel,
                         V::Array{PolyhedralFunction},
                         D::Array{PolyhedralFunction},
                         n_pass::Int,
                         seed::AbstractInitialSampler;
                         nprune::Int = n_pass,
                         pruner=nothing,
                         verbose::Int=n_pass+1,
                         l1_regularization::Real = 1e6)

    n_pruning = div(n_pass, nprune)
    println("** Dual SDDP with $(n_pass) passes, $(n_pruning) pruning  **")
    if n_pruning > 0 && isnothing(pruner)
        error("Could not proceed to pruning as `pruner` is not set.")
    end

    T, S = size(model.ξs)
    if isa(l1_regularization, Vector)
        m_dual = [dual_bellman_operator(model, t, l1_regularization[t]) for t in 1:T]
    else
        m_dual = [dual_bellman_operator(model, t, l1_regularization) for t in 1:T]
    end
    m_primal = [bellman_operator(model, t) for t in 1:T]

    for (t, Dₜ₊₁) in enumerate(D[2:end])
        initialize_lift_dual!(m_dual[t], model, t, Dₜ₊₁)
    end
    for (t, Vₜ₊₁) in enumerate(V[2:end])
        initialize_lift_primal!(m_primal[t], model, t, Vₜ₊₁)
    end

    println("Bellman JuMP Models initialized")
    println("Now running SDDP passes")

    for i in 1:n_pass
        ξscenarios = model.ξs[:,rand(1:S,1),:]

        x₀ = sample(seed)
        xscenarios = forward_pass(model, m_primal, ξscenarios, x₀)
        μscenarios = backward_pass!(model, m_primal, V, xscenarios)
        dual_backward_pass!(model, m_dual, D, μscenarios)
        l1_reg = isa(l1_regularization, Vector) ? l1_regularization[1] : l1_regularization
        μ₀ = initial_position(D[1], seed, pruner.optimizer_constructor, l1_reg)
        dual_cupps_pass(model, m_dual, ξscenarios, D, μ₀)

        if mod(i, nprune) == 0
            println("\n Performing pruning number $(div(i, nprune))")
            for (t, Dₜ₊₁) in enumerate(D[2:end])
                prune!(D[t+1], pruner)
                l1_reg = isa(l1_regularization, Vector) ? l1_regularization[t] : l1_regularization
                m_dual[t] = dual_bellman_operator(model, t, l1_reg)
                initialize_lift_dual!(m_dual[t], model, t, D[t+1])
            end
            V[1] = unique(V[1])
            for (t, Vₜ₊₁) in enumerate(V[2:end])
                V[t+1] = unique(Vₜ₊₁)
                prune!(V[t+1], pruner)
                m[t] = bellman_operator(hdm, t)
                initialize_lift_primal!(m[t], hdm, t, V[t+1])
            end
        end
        if mod(i, verbose) == 0
            l1_reg = isa(l1_regularization, Vector) ? l1_regularization[1] : l1_regularization
            v₀ = PolyhedralFenchelTransform(D[1], l1_reg)
            lb = V[1](x₀)
            ub = v₀(x₀, pruner.optimizer_constructor)
            println("Iter $i  lb ", lb, "  ub ", ub)
        end
    end
    return m_primal, m_dual
end
