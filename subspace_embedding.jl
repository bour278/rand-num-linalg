"""
    sparse_subspace_embedding(A::Matrix{Float64}, t::Int)

Create a sparse subspace embedding matrix S for the column space of A.
The embedding preserves lengths of all vectors in the column space.

Parameters:
- A: n×d matrix (n >> d)
- t: target embedding dimension

Returns:
- S*A: t×d matrix that is a subspace embedding of A
"""
function sparse_subspace_embedding(A::Matrix{Float64}, t::Int)
    n, d = size(A)
    
    if t < d * log(d)
        @warn "Embedding dimension $t may be too small; recommended minimum is $(d * log(d))"
    end
    
    S = spzeros(t, n)
    
    for j = 1:n
        h = rand(1:t)
        S[h, j] = rand() < 0.5 ? -1.0 : 1.0
    end
    
    return S * A
end

"""
    gaussian_subspace_embedding(A::Matrix{Float64}, k::Int)

Create a Gaussian random projection matrix for subspace embedding.
This is the Johnson-Lindenstrauss transform approach.

Parameters:
- A: n×d matrix
- k: target embedding dimension

Returns:
- R*A: k×d matrix that is a subspace embedding of A
"""
function gaussian_subspace_embedding(A::Matrix{Float64}, k::Int)
    n, d = size(A)
    
    R = randn(k, n) / sqrt(k)
    
    return R * A
end

using SparseArrays
using Random