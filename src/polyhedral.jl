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
