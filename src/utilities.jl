function variables_product(args...)
    dim_variable = length(args)
    products = Base.product(args...)
    products_array = fill(0., length(products), dim_variable)
    for (i, product) in enumerate(products)
        products_array[i,:] .= [product...]  
    end 
    products_array
end

function export_lbo_to_mps(m::JuMP.Model, filename::String)
    dest = MOI.FileFormats.Model(format = MOI.FileFormats.FORMAT_MPS)
    MOI.copy_to(dest, m)
    MOI.write_to_file(dest, filename)
end

function export_lbo_to_lp(m::JuMP.Model, filename::String)
    dest = MOI.FileFormats.Model(format = MOI.FileFormats.FORMAT_LP)
    MOI.copy_to(dest, m)
    MOI.write_to_file(dest, filename)
end

# returns the problem matrices in scipy linprog form
# min c'x 
# s.t A_ub x <= b_ub
#     A_eq x = b_eq
#     x_lb <= x <= x_ub
#
# plus the indices of the states to fix their value 
function export_lbo_to_scipylp(m::JuMP.Model)
    Asparse, lb, ub, _ = JuMP._std_matrix(m)
    nvars =  JuMP.num_variables(m)
    ncons = length(lb) - nvars

    x_lb = lb[1:nvars]
    x_ub = ub[1:nvars]
    indices_xₜ = [x.index.value for x in m[:xₜ]]

    indices_uₜ = [u.index.value for u in m[:uₜ₊₁]]

    # parse objective function to compute vector c such as std_obj = c'x
    objective_expr = objective_function(m)
    c = zeros(nvars)
    for pair in objective_expr.terms
        c[pair[1].index.value] = pair[2]
    end

    rows_ub = Int[]
    rows_lb = Int[]
    rows_eq = Int[]
    # parse constraints
    for (i,row) in enumerate(eachrow(Asparse))
        if lb[nvars + i] == ub[nvars + i] && ub[nvars + i] < Inf 
            push!(rows_eq, i)
        else
            if lb[nvars + i] > -Inf
                push!(rows_lb, i)
            end
            if ub[nvars + i] < Inf
                push!(rows_ub, i)
            end
        end 
    end

    A_ub = cat(Asparse[rows_ub,1:nvars], -Asparse[rows_lb,1:nvars], dims = 1)
    A_eq = Asparse[rows_eq,1:nvars]

    b_ub = cat(ub[nvars .+ rows_ub], -lb[nvars .+ rows_lb], dims = 1) 
    b_eq = lb[nvars .+ rows_eq]

    c, A_ub, b_ub, A_eq, b_eq, x_lb, x_ub, indices_xₜ.-1, indices_uₜ.-1
end
# using PyCall
# scipysparse = pyimport("scipy.sparse")
# np = pyimport("numpy")
# scipysparse.save_npz("A_eq.npz", scipysparse.csc_matrix(A_eq))
# scipysparse.save_npz("A_ub.npz", scipysparse.csc_matrix(A_ub))
# np.savez("vectors.npz", c = c, b_ub = b_ub, b_eq = b_eq, x_lb = x_lb, 
#                         x_ub = x_ub, indices_x = indices_x, 
#                         indices_u = indices_u)
