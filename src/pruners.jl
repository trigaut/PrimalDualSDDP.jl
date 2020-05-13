
abstract type AbstractPruner end

function prune!(V::PolyhedralFunction, pruner) end

struct ExactPruner <: AbstractPruner
    optimizer_constructor
    lb
    ub
end

function ExactPruner(optimizer_constructor; lb = -Inf, ub = Inf)
    ExactPruner(optimizer_constructor, lb, ub)
end

function prune!(V::PolyhedralFunction, pruner::ExactPruner)
    dim_V = dim(V)
    for (ind, cut) in Iterators.reverse(enumerate(eachcut(V)))
        λ, γ = cut
        lp = Model(pruner.optimizer_constructor)
        @variable(lp, x[1:dim_V])
        if pruner.ub != Inf
            @constraint(lp, x .<= pruner.ub)
        end
        if pruner.lb != -Inf
            @constraint(lp, x .>= pruner.lb)
        end
        @variable(lp, y)
        @objective(lp, Max, dot(λ, x) - y)
        for (C,γc) in eachcut(V)
            if C != λ
                @constraint(lp, dot(C, x) + γc <= y)
            end
        end
        optimize!(lp)
        if termination_status(lp) == MOI.OPTIMAL && objective_value(lp) + γ <= 0.
            remove_cut!(V, ind)
        end
    end
    return
end

struct CutShooter <: AbstractPruner
    shooter_position
end