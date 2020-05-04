module PrimalDualSDDP

using JuMP, Clp, Dualization, CutPruners
using LinearAlgebra, Statistics, MultivariateStats, Clustering 
using ProgressMeter

export HazardDecisionModel, DecisionHazardModel, PolyhedralFunction

abstract type LinearBellmanModel end

function bellman_operator end

function dual_bellman_operator end

include("statistics.jl")
include("utilities.jl")
include("polyhedral.jl")
include("sddp_dh.jl")
include("sddp_hd.jl")
include("dual_sddp.jl")
include("dualization.jl")

end # module
