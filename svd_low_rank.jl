"""
    svd_low_rank_approximation(A::Matrix{Float64}, k::Int)

Compute a low-rank approximation of matrix A using randomized sampling and SVD.

Parameters:
- A: m×n matrix to approximate
- k: target rank of approximation
- r: number of rows to sample (defaults to 2k)

Returns:
- low-rank approximation of A
"""
function svd_low_rank_approximation(A::Matrix{Float64}, k::Int; r::Int=2*k)
    m, n = size(A)
    
    row_norms_squared = vec(sum(A.^2, dims=2))
    A_F_squared = sum(row_norms_squared)
    row_probs = row_norms_squared / A_F_squared
    
    row_indices = sample(1:m, Weights(row_probs), r, replace=true)
    R = zeros(r, n)
    
    for t = 1:r
        i = row_indices[t]
        scaling_factor = 1.0 / sqrt(r * row_probs[i])
        R[t, :] = A[i, :] * scaling_factor
    end
    
    F = svd(R)
    
    V_k = F.V[:, 1:min(k, size(F.V, 2))]
    
    return A * V_k * V_k'
end

using StatsBase
using LinearAlgebra