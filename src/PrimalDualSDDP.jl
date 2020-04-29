module PrimalDualSDDP

using JuMP, Clp, Dualization, CutPruners, CPLEX
using LinearAlgebra, Statistics, MultivariateStats, Clustering 
using ProgressMeter

abstract type LinearBellmanModel end

include("statistics.jl")
include("utilities.jl")
include("polyhedral.jl")
include("sddp_dh.jl")
include("sddp_hd.jl")
include("dual_sddp.jl")
include("dualization.jl")

include("models/nim_dh.jl")
include("models/nim_net_dh.jl")
include("models/nim_net_hd.jl")
include("models/water_dam_hd.jl")

end # module
