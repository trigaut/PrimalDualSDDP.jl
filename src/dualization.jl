function auto_dual_bellman_operator(m::JuMP.Model, weights::Vector{Float64}, lip::Real)
    md = dualize(m; dual_names = DualNames("", ""))

    nx = length(m[:xₜ])
    info =  VariableInfo(false, NaN, false, NaN, false, NaN,
                         false, NaN, false, false)

    md[:μₜ] = VariableRef[]
    for var in name.(m[:xₜ])
        con = JuMP.constraint_by_name(md, var)
        adjointvar = add_variable(md, build_variable(error, info), "μ"*var)
        push!(md[:μₜ], adjointvar)
        set_normalized_coefficient(con, adjointvar, 1.)
    end

    xₜ₊₁uniquenames = unique(cat(map(x -> name.(x), m[:xₜ₊₁])..., dims=1))

    info =  VariableInfo(true, -lip, true, lip, false, NaN,
                         false, NaN, false, false)

    for var in xₜ₊₁uniquenames
        con = JuMP.constraint_by_name(md, var)
        adjointvar = add_variable(md, build_variable(error, info), "μ"*var)
        varindex = match(r".*\[(\d+)\]", var)
        if isnothing(varindex)
            set_normalized_coefficient(con, adjointvar, -sum(weights))
        else
            varindex = parse(Int, varindex.captures[1])
            set_normalized_coefficient(con, adjointvar, -weights[varindex])
        end
    end

    md[:μₜ₊₁] = Array{Array{VariableRef,1},1}()
    for var_array in m[:xₜ₊₁]
        push!(md[:μₜ₊₁], variable_by_name.(md, "μ".*name.(var_array)))
    end

    obj_expr = objective_function(md)
    @objective(md, Min, -obj_expr)

    return md
end

function auto_dual_bellman_operator(dhm::DecisionHazardModel, t::Int, lip::Real)
    m = bellman_operator(dhm, t)
    md = auto_dual_bellman_operator(m, dhm.πξ[t], lip)
    return md
end
