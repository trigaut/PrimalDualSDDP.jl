abstract type DecisionHazardModel<: LinearBellmanModel end

function initialize_lift_primal!(m::JuMP.Model,
                                 dhm::DecisionHazardModel,
                                 t::Int,
                                 Vₜ₊₁::PolyhedralFunction)
    
    nξ = length(m[:xₜ₊₁])
    @variable(m, θ[1:nξ])
    m.ext[:cuts] = Dict()
    for (λ, γ) in cuts(Vₜ₊₁)
        if !isbound((λ, γ))
            m.ext[:cuts][λ] = ConstraintRef[]
            for i in 1:nξ
                c = @constraint(m, θ[i] >= λ[1:end-1]'*m[:xₜ₊₁][i] + γ)
                push!(m.ext[:cuts][λ], c)
            end
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
                  mₜ::JuMP.Model, 
                  xₜ::Vector{Float64})
    primalsolve!(dhm, mₜ, xₜ)
    λ = dual.(FixRef.(mₜ[:xₜ]))
    γ = objective_value(mₜ) - λ'*xₜ
    cut = λ => γ
    push_cut!(Vₜ, cut)
    return cut
end

function update_cut_in_model!(dhm::DecisionHazardModel, 
                              mₜ::JuMP.Model, 
                              cut::Cut)
    nξ = length(mₜ[:θ])
    λ, γ = cut
    if λ in keys(mₜ.ext[:cuts])
        constraints = mₜ.ext[:cuts][λ]
        JuMP.set_normalized_rhs.(constraints, γ)
    else
        mₜ.ext[:cuts][λ] = ConstraintRef[]
        for i in 1:nξ
            c = @constraint(mₜ, mₜ[:θ][i] >= λ'*mₜ[:xₜ₊₁][i] + γ)
            push!(mₜ.ext[:cuts][λ], c)
        end
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
    for pass in 1:n_pass
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
                        V::Vector{PolyhedralFunction{R}},
                        xscenarios::Array{Float64,3}) where R <: Real
    T = length(m)
    _, n_pass, x_dim = size(xscenarios)
    λ = zeros(x_dim)
    for t in T:-1:2
        for pass in 1:n_pass
            cut = new_cut!(dhm, V[t], m[t], xscenarios[t,pass,:])
            update_cut_in_model!(dhm, m[t-1], cut)
        end
    end
    for pass in 1:n_pass
        new_cut!(dhm, V[1], m[1], xscenarios[1,pass,:])
    end
    return 
end

function primalsddp!(dhm::DecisionHazardModel, 
                     V::Array{PolyhedralFunction{R}, 1}, 
                     n_pass::Int, 
                     x₀s::Array;
                     nprune::Int = n_pass,
                     prunetol::Real = 0.) where R <: Real
    println("** Primal SDDP with $(n_pass) passes, $(div(n_pass,nprune)) pruning  **")
    T, S = size(dhm.ξs)
    m = [bellman_operator(dhm, t) for t in 1:T]
    for (t, Vₜ₊₁) in enumerate(V[2:end])
        initialize_lift_primal!(m[t], dhm, t, Vₜ₊₁)
    end
    println("Primal Bellman JuMP Models initialized")
    println("Now running sddp passes")
    @showprogress for i in 1:n_pass
        ξscenarios = dhm.ξs[:,rand(1:S,1),:]
        x₀ = [rand(x₀s)...]
        xscenarios = forward_pass(dhm, m, ξscenarios, x₀)
        backward_pass!(dhm, m, V, xscenarios)
        if mod(i,nprune) == 0
            println("\n Performing pruning number $(div(i,nprune))")
            for (t, Vₜ₊₁) in enumerate(V[2:end])
                lp_pruning!(Vₜ₊₁)
                m[t] = bellman_operator(dhm, t)
                initialize_lift_primal!(m[t], dhm, t, V[t+1])
            end
        end
    end
    lp_pruning!(V[1])
    return m
end
