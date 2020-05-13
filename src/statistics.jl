function fitar_cholesky(scenarios::Array{T, 2}; order::Int=1) where T
    ntime = size(scenarios, 1)
    @assert order < ntime

    # Define non-stationary AR model such that :
    #    X[t+1] = α[t] * X[t] + β[t] + σ[t] * N(0, 1)
    α = zeros(T, ntime-1, order)
    β = zeros(T, ntime-1)
    σ = zeros(T, ntime-1)

    for t in (order):ntime-1
        sol = llsq(collect(scenarios[t-order+1:t, :]'), scenarios[t+1, :])

        α[t, :] = sol[1:order]
        β[t] = sol[end]

        ypred =   scenarios[t-order+1:t, :]' * α[t, :]  .+ β[t]
        σ[t] = std(ypred - scenarios[t+1, :])
    end

    return α, β, σ
end

function fitar_ridge_reg(scenarios::Array{T, 2}; order::Int=1) where T
    ntime = size(scenarios, 1)
    @assert order < ntime
    α = zeros(T, ntime-1, order)

    for t in (order):ntime-1
        sol = ridge(collect(scenarios[t-order+1:t, :]'), scenarios[t+1, :], 0.1, bias = false)
        α[t, :] = sol[1:order]
    end

    return α
end

function fitar_linear_reg(scenarios::Array{T, 2}; order::Int=1) where T
    ntime = size(scenarios, 1)
    @assert order < ntime

    # Define non-stationary AR model such that :
    #    X[t+1] = α[t] * X[t] + σ[t] * N(0, 1)
    α = zeros(T, ntime-1, order)

    for t in (order):ntime-1
        sol = scenarios[t-order+1:t, :]' \ scenarios[t+1, :]
        α[t, :] = sol[1:order]
    end

    return α
end

function discrete_white_noise(scenarios::Array{Float64}, support_size::Int)

    T = size(scenarios, 1)
    supports = []
    probas = []

    for t in 1:T
        R = kmeans(collect(scenarios[t,:,:]'), support_size)
        saved = R.counts .> 1e-6
        push!(supports, collect(R.centers[:,saved]'))
        push!(probas, R.counts[saved]./sum(R.counts[saved]))
    end

    supports, probas
end
