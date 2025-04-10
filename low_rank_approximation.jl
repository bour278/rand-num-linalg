"""
    low_rank_approximation(A::Matrix{Float64}, k::Int, r::Int, s::Int)

Compute a low-rank approximation of matrix A using column/row sampling.

Parameters:
- A: m×n matrix to approximate
- k: target rank of approximation
- r: number of rows to sample
- s: number of columns to sample

Returns:
- C*U*R: rank-k approximation of A
"""
function low_rank_approximation(A::Matrix{Float64}, k::Int, r::Int, s::Int)
    m, n = size(A)
    
    col_norms_squared = vec(sum(A.^2, dims=1))
    A_F_squared = sum(col_norms_squared)
    col_probs = col_norms_squared / A_F_squared
    
    col_indices = sample(1:n, Weights(col_probs), s, replace=true)
    C = zeros(m, s)
    
    for t = 1:s
        j = col_indices[t]
        scaling_factor = 1.0 / sqrt(s * col_probs[j])
        C[:, t] = A[:, j] * scaling_factor
    end
    
    row_norms_squared = vec(sum(A.^2, dims=2))
    row_probs = row_norms_squared / A_F_squared
    
    row_indices = sample(1:m, Weights(row_probs), r, replace=true)
    R = zeros(r, n)
    
    for t = 1:r
        i = row_indices[t]
        scaling_factor = 1.0 / sqrt(r * row_probs[i])
        R[t, :] = A[i, :] * scaling_factor
    end
    
    F_C = svd(C)
    V_C = F_C.V[:, 1:min(k, size(F_C.V, 2))]
    
    F_R = svd(R)
    U_R = F_R.U[:, 1:min(k, size(F_R.U, 2))]
    
    W = C' * A * R'
    
    U = V_C * pinv(V_C' * W * U_R) * U_R'
    
    return C * U * R
end

using StatsBase
using LinearAlgebra