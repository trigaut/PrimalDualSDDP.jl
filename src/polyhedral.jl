mutable struct PolyhedralFunction
	λ::Array{Float64,2}
	γ::Array{Float64,1}
end

ncuts(V::PolyhedralFunction) = size(V.λ, 1)
dim(V::PolyhedralFunction) = size(V.λ, 2)
eachcut(V::PolyhedralFunction) = zip(eachrow(V.λ), V.γ)

function PolyhedralFunction()
	return PolyhedralFunction(Array{Float64,2}(undef, 0,0), Vector{Float64}())
end

function (V::PolyhedralFunction)(x::Array{Float64,1})
	maximum(V.λ*x .+ V.γ)
end

function Base.unique(V::PolyhedralFunction)
	Vu = unique(cat(V.λ, V.γ, dims=2), dims=1)
	return PolyhedralFunction(Vu[:,1:end-1], Vu[:,end])
end

function remove_cut(V::PolyhedralFunction, cut_index::Int)
	PolyhedralFunction(V.λ[(1:cut_index-1)∪(cut_index+1:end),:], V.γ[(1:cut_index-1)∪(cut_index+1:end)])
end

function fenchel_transform(D::PolyhedralFunction, 
						   x::Array{Float64, 1}, 
						   lip::Float64)
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

function regularized_fenchel_transform(D::PolyhedralFunction, 
									   x::Array{Float64, 1}, 
									   lip::Float64)
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