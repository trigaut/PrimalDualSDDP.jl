mutable struct PolyhedralFunction
    λ::Array{Float64,2}
    γ::Array{Float64,1}
end

mutable struct PolyhedralFenchelTransform
    V::PolyhedralFunction
    l1_regularization::Real
end

ncuts(V::PolyhedralFunction) = size(V.λ, 1)
dim(V::PolyhedralFunction) = size(V.λ, 2)
eachcut(V::PolyhedralFunction) = zip(eachrow(V.λ), V.γ)

function PolyhedralFunction()
    return PolyhedralFunction(Array{Float64,2}(undef, 0,0), Vector{Float64}())
end

function lipschitz_constant(V::PolyhedralFunction, pnorm::Real = 1)
    return maximum([norm(λ,pnorm) for λ in eachrow(V.λ)])
end

function PolyhedralFenchelTransform(V::PolyhedralFunction)
    PolyhedralFenchelTransform(V, 0.)
end

function (V::PolyhedralFunction)(x::Array{Float64,1})
        return maximum(V.λ*x .+ V.γ)
end

function (D::PolyhedralFenchelTransform)(x::Array{Float64,1})
    return fenchel_transform_as_sup(D.V, x, D.l1_regularization)
end

function Base.unique(V::PolyhedralFunction)
    Vu = unique(cat(V.λ, V.γ, dims=2), dims=1)
    return PolyhedralFunction(Vu[:,1:end-1], Vu[:,end])
end

function add_cut!(V, λ, γ)
    if ncuts(V) > 0
        V.λ = cat(V.λ, λ', dims = 1)
        push!(V.γ, γ)
    else
        V.λ = λ'
        V.γ = [γ]
    end
    return
end

function remove_cut!(V::PolyhedralFunction, cut_index::Int)
    V.λ = V.λ[(1:cut_index-1)∪(cut_index+1:end),:] 
    V.γ = V.γ[(1:cut_index-1)∪(cut_index+1:end)]
    return
end


function remove_cut(V::PolyhedralFunction, cut_index::Int)
    return PolyhedralFunction(V.λ[(1:cut_index-1)∪(cut_index+1:end),:], 
                              V.γ[(1:cut_index-1)∪(cut_index+1:end)])
end

function δ(point::Vector{Float64}, reg::Float64)
    nx = length(point)
    V = PolyhedralFunction()
    for (i,p) in enumerate(point)
        add_cut!(V, [(i==j)*reg for j in 1:nx], p)
        add_cut!(V, [-(i==j)*reg for j in 1:nx], p)
    end
    return V
end

function fenchel_transform_as_sup(D::PolyhedralFunction, 
                                  x::Array{Float64, 1}, 
                                  lip::Real)
    m = Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))
    nx = size(D.λ, 2)
    @variable(m, -lip <= λ[1:nx] <= lip)
    @variable(m, θ)
    for (xk, βk) in eachcut(D)
        @constraint(m, θ >= xk'*λ + βk)
    end
    @objective(m, Max, x'*λ - θ)
    optimize!(m)
    return objective_value(m)
end

function fenchel_transform_as_inf(D::PolyhedralFunction, 
                                  x::Array{Float64, 1}, 
                                  lip::Real)
    m = Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))
    nc, nx = size(D.λ)
    @variable(m, σ[1:nc] >= 0)
    @constraint(m, sum(σ) == 1)
    @variable(m, y[1:nx])
    @variable(m, lift[1:nx] >= 0)
    @constraint(m, lift .>= x .- y)
    @constraint(m, lift .>= y .- x)
    @constraint(m, sum(σk .* D.λ[k,:] for (k,σk) in enumerate(σ)) .== y)
    @objective(m, Min, lip*sum(lift) - sum(σ.*D.γ))
    optimize!(m)
    return objective_value(m)
end

function exact_pruning!(V::PolyhedralFunction; 
                        lb::Vector = fill(-1e6, dim(V) + 1), 
                        ub::Vector = fill(1e6, dim(V) + 1),
                        ϵ::Real = 0.)
    islb = true # max of cuts
    isfun = true # max of cuts
    optimizer_constructor = optimizer_with_attributes(Clp.Optimizer, 
                                                      "LogLevel" => 0)

    dominated_cuts = CutPruners.getdominated(V.λ, V.γ, islb, isfun, 
                                             optimizer_constructor, 
                                             lb, ub, ϵ)

    for cut_index in reverse(dominated_cuts)
        remove_cut!(V, cut_index)
    end
    return
end

function exact_pruning(V::PolyhedralFunction;
                       lb::Vector = fill(-Inf, dim(V) + 1), 
                       ub::Vector = fill(Inf, dim(V) + 1 ),
                       ϵ::Real = 0.)
    V1 = unique(V)
    islb = true # max of cuts
    isfun = true # max of cuts
    optimizer_constructor = optimizer_with_attributes(Clp.Optimizer, 
                                                      "LogLevel" => 0)
    # optimizer_constructor = optimizer_with_attributes(CPLEX.Optimizer, 
    #                                                   "CPX_PARAM_SCRIND" => 0)

    dominated_cuts = CutPruners.getdominated(V1.λ, V1.γ, islb, isfun, 
                                             optimizer_constructor, 
                                             lb, ub, ϵ)

    for cut_index in reverse(dominated_cuts)
        remove_cut!(V1, cut_index)
    end
    return V1
end
