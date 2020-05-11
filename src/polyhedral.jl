struct PolyhedralFunction{T <: Real} 
    halfspaces::Dict{Vector{T}, T}
end

function PolyhedralFunction{T}() where T <: Real
    return PolyhedralFunction{T}(Dict{Vector{T}, T}())
end

ndims(f::PolyhedralFunction) = size(first(f.halfspaces)[1],1)-1
Base.getindex(f::PolyhedralFunction, λ::Vector) = f.halfspaces[λ]
Base.setindex!(f::PolyhedralFunction, γ::Real, λ::Vector) = f.halfspaces[λ] = γ
slopes(f::PolyhedralFunction) = keys(f.halfspaces)
cuts(f::PolyhedralFunction) = f.halfspaces
ncuts(f::PolyhedralFunction) = f.halfspaces.count

Cut{T} = Union{Pair{Array{T,1},T}, Tuple{Array{T,1},T}} where T <: Real

function push_halfspace!(f::PolyhedralFunction{T}, halfspace::Cut{T}) where T <: Real
    λ = halfspace[1]
    γ = halfspace[2]
    if λ ∉ slopes(f) || f[λ] < γ
        f[λ] = γ 
    end
    return
end

function push_cut!(f::PolyhedralFunction{T}, cut::Cut{T}) where T <: Real
    λ = vcat(cut[1], -1)
    γ = cut[2]
    if λ ∉ slopes(f) || f[λ] < γ
        f[λ] = γ 
    end
    return
end

function remove_cut!(f::PolyhedralFunction, slope::Vector)
    delete!(f.halfspaces, slope)
end

function remove_cut!(f::PolyhedralFunction, cut::Cut)
    λ = cut[1]
    remove_cut!(f, λ)
end

function PolyhedralFunction(arg::Cut{T}, args...) where T <: Real
    f = PolyhedralFunction{T}()
    push_cut!(f, arg)
    for cut in args
        push_cut!(f, cut)
    end
    return f
end

function update_bounds!(f::PolyhedralFunction{T}, dim::Int; 
                     lower_bound::Real = -Inf,  upper_bound::Real = Inf) where T <: Real
    if lower_bound > -Inf
        d = ndims(f)+1
        λ = zeros(T, d)
        λ[dim] = -1
        γ = lower_bound
        f.halfspaces[λ] = γ
    end
    if upper_bound < Inf
        d = ndims(f)+1
        λ = zeros(T, d)
        λ[dim] = 1
        γ = -upper_bound
        f.halfspaces[λ] = γ
    end
    return
end

function isbound(cut::Cut)
    cut[1][end] == 0
end

function (f::PolyhedralFunction{T})(x::Vector{T}) where T <: Real
    result = -Inf
    for (λ, γ) in cuts(f)
        if !isbound((λ,γ))
            result = max(dot(λ[1:end-1],x) + γ, result)
        end
    end
    result
end

struct PolyhedralFenchelTransform{T}
    V::PolyhedralFunction
    l1_regularization::Real
end

function lipschitz_constant(V::PolyhedralFunction, pnorm::Real = 1)
    return maximum([norm(λ,pnorm) for λ in slopes(V)])
end

function PolyhedralFenchelTransform(V::PolyhedralFunction)
    PolyhedralFenchelTransform(V, 0.)
end

function (D::PolyhedralFenchelTransform)(x::Array{Float64,1})
    return fenchel_transform_as_sup(D.V, x, D.l1_regularization)
end

function δ(point::Vector{T}, reg::Float64) where T <: Real
    nx = length(point)
    V = PolyhedralFunction{T}()
    for (i,p) in enumerate(point)
        push_cut!(V, ([(i==j)*reg for j in 1:nx], p))
        push_cut!(V, ([-(i==j)*reg for j in 1:nx], p))
    end
    return V
end

function fenchel_transform_as_sup(D::PolyhedralFunction, 
                                  x::Array{Float64, 1}, 
                                  lip::Real)
    m = Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))
    nx = ndims(D)
    @variable(m, -lip <= λ[1:nx] <= lip)
    @variable(m, θ)
    for (xk, βk) in cuts(D)
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
    nc = ncuts(D)
    nx = ndims(D)
    @variable(m, σ[1:nc] >= 0)
    @constraint(m, sum(σ) == 1)
    @variable(m, y[1:nx])
    @variable(m, lift[1:nx] >= 0)
    @constraint(m, lift .>= x .- y)
    @constraint(m, lift .>= y .- x)
    @constraint(m, sum(σ[k] .* λk for (k,λk) in enumerate(slopes(D))) .== y)
    @objective(m, Min, lip*sum(lift) - sum(σ[k] .* D[λk] for (k,λk) in enumerate(slopes(D))))
    optimize!(m)
    return objective_value(m)
end

function intersection_list(xi::Vector, d::Vector, f::PolyhedralFunction)
    distances = Float64[]
    hit_cuts = []
    for (C, γc) in cuts(f)
        Cxd = dot(C,d)
        if Cxd != 0.
            tc = (-γc - dot(C,xi)) / Cxd
            if tc > 0
                push!(distances, tc)
                push!(hit_cuts, (C, γc))
            end
        end
    end
    
    # Improve the following ?
    cuts_order = sortperm(distances)
    dmin = distances[cuts_order[1]]
    closest_cuts = [hit_cuts[cuts_order[1]]]
    for ind in cuts_order[2:end]
        if distances[ind] == dmin
            push!(closest_cuts, hit_cuts[ind])
        else
            break
        end
    end
    closest_cuts
end

function raytracing_pruning!(fref::PolyhedralFunction{T}, xi::Vector) where T <: Real
    f = deepcopy(fref)
    fm = PolyhedralFunction{T}()
    dim_f = ndims(f)
    Is = Dict()
    println("First Phase")
    for (Ci, γ) in cuts(f)
        I = intersection_list(xi, Ci, fref)
        if size(I,1) == 1
            push_halfspace!(fm, I[1])
            remove_cut!(f, I[1][1])
            println("Saving cut: ", I[1], " using vector: ", Ci)
        end
        Is[Ci] = I
    end
    for (Ci, γ) in cuts(f)
        remove_cut!(f, Ci)
        lp = Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))
        @variable(lp, x[1:dim_f+1])
        @objective(lp, Max, dot(Ci, x))
        for (C,γc) in cuts(f)
            @constraint(lp, dot(C, x) + γc <= 0.)
        end
        for (C,γc) in cuts(fm)
            @constraint(lp, dot(C, x) + γc <= 0.)
        end
        optimize!(lp)
        if (termination_status(lp) != MOI.OPTIMAL) || (objective_value(lp) + γ > 0.)
            push_halfspace!(fm, (Ci,γ))
        end
    end
    fm
end

function lp_pruning(fref::PolyhedralFunction{T}) where T <: Real
    dim_f = ndims(fref)
    f = deepcopy(fref)
    fm = PolyhedralFunction{T}()
    for (Ci, γ) in cuts(f)
        remove_cut!(f, Ci)
        lp = Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))
        @variable(lp, x[1:dim_f+1])
        @objective(lp, Max, dot(Ci, x))
        for (C,γc) in cuts(f)
            @constraint(lp, dot(C, x) + γc <= 0.)
        end
        for (C,γc) in cuts(fm)
            @constraint(lp, dot(C, x) + γc <= 0.)
        end
        optimize!(lp)
        if (termination_status(lp) != MOI.OPTIMAL) || (objective_value(lp) + γ > 0.)
            push_halfspace!(fm, (Ci,γ))
        end
    end
    fm
end