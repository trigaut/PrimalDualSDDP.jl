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