"""
    length_squared_sampling(A::Matrix{Float64}, B::Matrix{Float64}, s::Int)

Approximate the matrix product A*B by sampling s columns of A and corresponding rows of B
according to the length-squared distribution.

Parameters:
- A: m×n matrix
- B: n×p matrix
- s: number of samples

Returns:
- C*R: approximation of A*B
"""
function length_squared_sampling(A::Matrix{Float64}, B::Matrix{Float64}, s::Int)
    m, n = size(A)
    if size(B, 1) != n
        error("Matrix dimensions must match for multiplication")
    end
    p = size(B, 2)
    
    col_norms_squared = vec(sum(A.^2, dims=1))
    A_F_squared = sum(col_norms_squared)
    probs = col_norms_squared / A_F_squared
    
    indices = sample(1:n, Weights(probs), s, replace=true)
    
    C = zeros(m, s)
    R = zeros(s, p)
    
    for t = 1:s
        j = indices[t]
        scaling_factor = 1.0 / sqrt(s * probs[j])
        C[:, t] = A[:, j] * scaling_factor
        R[t, :] = B[j, :] * scaling_factor
    end
    
    return C * R
end

using StatsBase