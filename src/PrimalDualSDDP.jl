module PrimalDualSDDP

using JuMP, Dualization
using LinearAlgebra, Statistics, MultivariateStats, Clustering
using ProgressMeter

export HazardDecisionModel, DecisionHazardModel, PolyhedralFunction

abstract type LinearBellmanModel end

function bellman_operator end

function dual_bellman_operator end

include("statistics.jl")
include("utilities.jl")
include("polyhedral.jl")
include("pruners.jl")
include("sddp_dh.jl")
include("sddp_hd.jl")
include("dual_sddp.jl")
include("dualization.jl")
include("simulation_hd.jl")

end # module
