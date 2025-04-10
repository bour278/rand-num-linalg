"""
    spectral_norm_sampling(A::Matrix{Float64}, s::Int, ε::Float64)

Sample columns of A to create a matrix C such that ||CC' - AA'||₂ ≤ ε||A||₂||A||_F
using the Rudelson-Vershynin bound.

Parameters:
- A: m×n matrix
- s: number of samples
- ε: error threshold

Returns:
- C: m×s matrix such that CC' approximates AA'
"""
function spectral_norm_sampling(A::Matrix{Float64}, s::Int, ε::Float64)
    m, n = size(A)
    
    A_spectral = opnorm(A, 2)
    A_frobenius = norm(A)
    
    min_s = ceil(Int, log(m) / ε^2)
    if s < min_s
        @warn "Sample size $s may be too small; recommended minimum is $min_s"
    end
    
    col_norms_squared = vec(sum(A.^2, dims=1))
    A_F_squared = sum(col_norms_squared)
    probs = col_norms_squared / A_F_squared
    
    indices = sample(1:n, Weights(probs), s, replace=true)
    
    C = zeros(m, s)
    
    for t = 1:s
        j = indices[t]
        scaling_factor = 1.0 / sqrt(s * probs[j])
        C[:, t] = A[:, j] * scaling_factor
    end
    
    return C
end

using StatsBase
using LinearAlgebra