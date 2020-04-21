module PrimalDualSDDP

using JuMP, Clp, Statistics, MultivariateStats, Clustering, ProgressMeter, Dualization

include("statistics.jl")
include("utilities.jl")
include("polyhedral.jl")
include("sddp_dh.jl")
# include("sddp_hd.jl")
include("dualization.jl")

include("models/nim_dh.jl")
include("models/nim_net_dh.jl")
include("models/water_dam_hd.jl")

end # module
