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


function dual_forward_pass(lbm::LinearBellmanModel,
                           m::Vector{JuMP.Model}, 
                           ξscenarios::Array{Float64, 3},
                           μ₀::Vector{Float64})
    T = size(ξscenarios,1)
    n_pass = size(ξscenarios,2)
    μscenarios = fill(0., T+1, n_pass, length(μ₀))
    for pass in 1:n_pass
        μₜ = μ₀ 
        μscenarios[1,pass,:] .= μ₀
        for (t, ξₜ₊₁) in enumerate(eachrow(ξscenarios[:,pass,:]))
            μₜ₊₁ = dualstate!(lbm, m[t], μₜ)
            μscenarios[t+1,pass,:] .= rand(μₜ₊₁)
            μₜ = μscenarios[t+1,pass,:]
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
    for t in T:-1:1
        dualupdate!(lbm, m[t], D[t+1])
        for pass in 1:n_pass
            dual_new_cut!(lbm, D[t], m[t], μscenarios[t,pass,:])
        end
    end
    return 
end

function dualsddp!(lbm::LinearBellmanModel, 
                    D::Array{PolyhedralFunction}, 
                    n_pass::Int, 
                    μ₀s::Array;
                    nprune::Int = n_pass,
                    l1_regularization::Real = 1e6)
    println("** Dual SDDP with $(n_pass) passes, $(div(n_pass,nprune)) pruning  **")
    T, S = size(lbm.ξs)
    m = [dual_bellman_operator(lbm, t, l1_regularization) for t in 1:T]
    for (t, Dₜ₊₁) in enumerate(D[2:end])
        initialize_lift_dual!(m[t], lbm, t, Dₜ₊₁)
    end
    println("Dual Bellman JuMP Models initialized")
    println("Now running sddp passes")
    @showprogress for i in 1:n_pass
        ξscenarios = lbm.ξs[:,rand(1:S,1),:]
        μ₀ = [rand(μ₀s)...]
        μscenarios = dual_forward_pass(lbm, m, ξscenarios, μ₀)
        dual_backward_pass!(lbm, m, D, μscenarios)
        if mod(i,nprune) == 0
            println("\n Performing pruning number $(div(i,nprune))")
            D[1] = unique(D[1])
            for (t, Dₜ₊₁) in enumerate(D[2:end])
                D[t+1] = unique(Dₜ₊₁)
                m[t] = dual_bellman_operator(lbm, t, l1_regularization)
                initialize_lift_dual!(m[t], lbm, t, D[t+1])
            end
        end
    end
    return m
end