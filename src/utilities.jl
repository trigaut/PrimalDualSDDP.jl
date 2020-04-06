function variables_product(args...)
	dim_variable = length(args)
	products = Base.product(args...)
	products_array = fill(0., length(products), dim_variable)
	for (i, product) in enumerate(products)
		products_array[i,:] .= [product...]  
	end 
	products_array
end