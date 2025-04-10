"""
    tensor_approximation(A::Array{Float64}, k::Int, ε::Float64)

Compute a low-rank approximation of a tensor A using length-squared sampling.

Parameters:
- A: r-dimensional tensor
- k: maximum number of rank-1 tensors in approximation
- ε: error threshold

Returns:
- Sum of rank-1 tensors approximating A
"""
function tensor_approximation(A::Array{Float64}, k::Int, ε::Float64)
    dims = size(A)
    r = length(dims)
    
    result = zeros(dims...)
    frobenius_norm_squared = sum(A.^2)
    
    for i = 1:k
        current_tensor = A - result
        current_norm_squared = sum(current_tensor.^2)
        
        if current_norm_squared <= ε^2 * frobenius_norm_squared
            break
        end
        
        vectors = find_maximizing_vectors(current_tensor, r, ε)
        
        coefficient = tensor_multilinear_form(current_tensor, vectors)
        
        rank1_tensor = outer_product(vectors, coefficient)
        result = result + rank1_tensor
    end
    
    return result
end

"""
    find_maximizing_vectors(A::Array{Float64}, r::Int, ε::Float64)

Find vectors that approximately maximize the multilinear form defined by tensor A.
Uses length-squared sampling to estimate the tensor-vector products.
"""
function find_maximizing_vectors(A::Array{Float64}, r::Int, ε::Float64)
    dims = size(A)
    vectors = [randn(dims[i]) for i = 1:r]
    
    vectors = [v / norm(v) for v in vectors]
    
    for iteration = 1:10
        for dim = 1:r
            fixed_vectors = vectors[1:end .!= dim]
            
            new_vector = compute_mode_vector(A, dim, fixed_vectors, ε)
            
            vectors[dim] = new_vector / norm(new_vector)
        end
    end
    
    return vectors
end

"""
    compute_mode_vector(A::Array{Float64}, mode::Int, fixed_vectors::Vector, ε::Float64)

Compute the vector for a specific mode that maximizes the multilinear form,
given fixed vectors for other modes. Uses length-squared sampling.
"""
function compute_mode_vector(A::Array{Float64}, mode::Int, fixed_vectors::Vector, ε::Float64)
    dims = size(A)
    result = zeros(dims[mode])
    
    mode_matrix = reshape_mode(A, mode)
    
    s = ceil(Int, 10 / ε^2)
    
    flattened_tensor = reshape(A, :)
    probs = flattened_tensor.^2 / sum(flattened_tensor.^2)
    
    indices = sample(1:length(flattened_tensor), Weights(probs), s, replace=true)
    
    for idx in indices
        tensor_idx = CartesianIndices(dims)[idx].I
        
        value = A[tensor_idx...]
        for d = 1:length(dims)
            if d != mode
                vector_idx = d < mode ? d : d - 1
                value *= fixed_vectors[vector_idx][tensor_idx[d]]
            end
        end
        
        result[tensor_idx[mode]] += value / (s * probs[idx])
    end
    
    return result
end

"""
    reshape_mode(A::Array{Float64}, mode::Int)

Reshape tensor A to a matrix where the specified mode is the rows
and all other modes are flattened to columns.
"""
function reshape_mode(A::Array{Float64}, mode::Int)
    dims = size(A)
    mode_dim = dims[mode]
    other_dims = prod(dims[1:end .!= mode])
    
    perm = [mode; setdiff(1:length(dims), mode)]
    B = permutedims(A, perm)
    
    return reshape(B, mode_dim, other_dims)
end

"""
    tensor_multilinear_form(A::Array{Float64}, vectors::Vector)

Compute the multilinear form value for tensor A and vectors.
"""
function tensor_multilinear_form(A::Array{Float64}, vectors::Vector)
    result = 0.0
    dims = size(A)
    
    for idx in CartesianIndices(dims)
        term = A[idx]
        for d = 1:length(dims)
            term *= vectors[d][idx[d]]
        end
        result += term
    end
    
    return result
end

"""
    outer_product(vectors::Vector, coefficient::Float64)

Compute the outer product of vectors scaled by coefficient.
"""
function outer_product(vectors::Vector, coefficient::Float64)
    dims = [length(v) for v in vectors]
    result = zeros(dims...)
    
    # Create array with coordinates for each dimension
    indices = [1:d for d in dims]
    
    # Iterate over all combinations of indices
    for idx in Iterators.product(indices...)
        value = coefficient
        for d = 1:length(dims)
            value *= vectors[d][idx[d]]
        end
        result[idx...] = value
    end
    
    return result
end

using StatsBase
using LinearAlgebra