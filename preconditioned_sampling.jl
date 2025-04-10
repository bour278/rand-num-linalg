"""
    preconditioned_sampling(A::Matrix{Float64}, k::Int, s::Int)

Perform preconditioned length-squared sampling for low-rank approximation.
Samples according to leverage scores (column norms of A⁺A).

Parameters:
- A: m×n matrix
- k: target rank of approximation
- s: number of samples

Returns:
- C*X: rank-k approximation of A
"""
function preconditioned_sampling(A::Matrix{Float64}, k::Int, s::Int)
    m, n = size(A)
    
    F = svd(A)
    
    V_k = F.V[:, 1:min(k, length(F.S))]
    leverage_scores = vec(sum(V_k.^2, dims=2))
    
    probs = leverage_scores / sum(leverage_scores)
    indices = sample(1:n, Weights(probs), s, replace=true)
    
    C = zeros(m, s)
    
    for t = 1:s
        j = indices[t]
        scaling_factor = 1.0 / sqrt(s * probs[j])
        C[:, t] = A[:, j] * scaling_factor
    end
    
    X = pinv(C'*C) * C' * A
    
    return C * X
end

using StatsBase
using LinearAlgebra