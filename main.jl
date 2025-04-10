using LinearAlgebra
using StatsBase
using SparseArrays
using Random
using BenchmarkTools
using Plots
using Statistics

include("subspace_embedding.jl")
include("preconditioned_sampling.jl")
include("spectral_norm_error.jl")
include("tensor_approximation.jl")
include("svd_low_rank.jl")
include("low_rank_approximation.jl")
include("length_squared_sampling.jl")

function main()
    Random.seed!(42)  # For reproducibility
    
    println("Testing Randomized Linear Algebra Algorithms")
    println("============================================")
    
    n = 1000
    d = 20
    
    println("\nGenerating test matrices...")
    A = randn(n, d)  # Tall matrix
    B = randn(d, n)  # Wide matrix
    C = randn(d, d)  # Square matrix
    
    println("Matrix A: $n × $d")
    println("Matrix B: $d × $n")
    println("Matrix C: $d × $d")
    
    println("\n1. Testing Subspace Embedding")
    println("---------------------------")
    
    t = ceil(Int, d * log(d) * 2)
    
    time_sparse = @elapsed S_A = sparse_subspace_embedding(A, t)
    time_gaussian = @elapsed G_A = gaussian_subspace_embedding(A, t)
    
    println("Sparse embedding time: $(round(time_sparse, digits=5)) seconds")
    println("Gaussian embedding time: $(round(time_gaussian, digits=5)) seconds")
    
    println("Original matrix size: $(size(A))")
    println("Embedded matrix size: $(size(S_A))")
    
    println("\n2. Testing Length-Squared Sampling")
    println("--------------------------------")
    
    s_samples = 100
    time_ls = @elapsed AB_approx = length_squared_sampling(A, B, s_samples)
    time_exact = @elapsed AB_exact = A * B
    
    error_ratio = norm(AB_approx - AB_exact) / norm(AB_exact)
    
    println("Length-squared sampling time: $(round(time_ls, digits=5)) seconds")
    println("Exact multiplication time: $(round(time_exact, digits=5)) seconds")
    println("Relative error: $(round(error_ratio, digits=5))")
    
    println("\n3. Testing Low-Rank Approximation")
    println("-------------------------------")
    
    k_rank = 10
    r_samples = 2*k_rank
    s_samples = 2*k_rank
    
    time_svd_exact = @elapsed begin
        F = svd(A)
        A_k_exact = F.U[:, 1:k_rank] * Diagonal(F.S[1:k_rank]) * F.Vt[1:k_rank, :]
    end
    
    time_svd_rand = @elapsed A_k_svd = svd_low_rank_approximation(A, k_rank)
    time_lowrank = @elapsed A_k_lowrank = low_rank_approximation(A, k_rank, r_samples, s_samples)
    time_precond = @elapsed A_k_precond = preconditioned_sampling(A, k_rank, s_samples)
    
    error_svd = norm(A_k_svd - A_k_exact) / norm(A_k_exact)
    error_lowrank = norm(A_k_lowrank - A_k_exact) / norm(A_k_exact)
    error_precond = norm(A_k_precond - A_k_exact) / norm(A_k_exact)
    
    println("Exact SVD time: $(round(time_svd_exact, digits=5)) seconds")
    println("Randomized SVD time: $(round(time_svd_rand, digits=5)) seconds")
    println("Low-rank approximation time: $(round(time_lowrank, digits=5)) seconds")
    println("Preconditioned sampling time: $(round(time_precond, digits=5)) seconds")
    
    println("Randomized SVD error: $(round(error_svd, digits=5))")
    println("Low-rank approximation error: $(round(error_lowrank, digits=5))")
    println("Preconditioned sampling error: $(round(error_precond, digits=5))")
    
    println("\n4. Testing Spectral Norm Error")
    println("---------------------------")
    
    s_samples = 100
    epsilon = 0.1
    
    time_spectral = @elapsed C_spectral = spectral_norm_sampling(A, s_samples, epsilon)
    
    AA_exact = A * A'
    CC_approx = C_spectral * C_spectral'
    
    spectral_error = opnorm(CC_approx - AA_exact) / opnorm(A) / norm(A)
    
    println("Spectral norm sampling time: $(round(time_spectral, digits=5)) seconds")
    println("Relative spectral error: $(round(spectral_error, digits=5))")
    println("Target error bound: $epsilon")
    
    println("\n5. Testing Tensor Approximation")
    println("----------------------------")
    
    tensor_size = 20
    tensor_dim = 3
    k_tensor = 5
    eps_tensor = 0.1
    
    T = rand(Float64, [tensor_size for _ in 1:tensor_dim]...)
    
    time_tensor = @elapsed T_approx = tensor_approximation(T, k_tensor, eps_tensor)
    
    tensor_error = sqrt(sum((T_approx - T).^2)) / sqrt(sum(T.^2))
    
    println("Tensor approximation time: $(round(time_tensor, digits=5)) seconds")
    println("Relative tensor error: $(round(tensor_error, digits=5))")
    
    println("\n6. Comparative Performance Visualization")
    println("-------------------------------------")
    
    time_data = [time_sparse, time_gaussian, time_ls, time_exact, 
                time_svd_exact, time_svd_rand, time_lowrank, time_precond, 
                time_spectral, time_tensor]
    
    algorithm_names = ["Sparse Embed", "Gaussian Embed", "Length² Sampling", "Exact Mult.", 
                      "Exact SVD", "Random SVD", "Low-rank", "Precond. Sampling", 
                      "Spectral Norm", "Tensor Approx."]
    
    p1 = bar(algorithm_names, time_data, 
        title = "Algorithm Run Times", 
        xlabel = "Algorithm", 
        ylabel = "Time (seconds)",
        legend = false,
        rotation = 45,
        xtickfontsize = 8,
        margin = 10Plots.mm)
    
    savefig(p1, "runtime_comparison.png")
    
    error_data = [error_ratio, error_svd, error_lowrank, error_precond, spectral_error, tensor_error]
    error_names = ["Length² Sampling", "Random SVD", "Low-rank", "Precond. Sampling", "Spectral Norm", "Tensor Approx."]
    
    p2 = bar(error_names, error_data, 
        title = "Approximation Errors", 
        xlabel = "Algorithm", 
        ylabel = "Relative Error",
        legend = false,
        rotation = 45,
        xtickfontsize = 8,
        margin = 10Plots.mm)
    
    savefig(p2, "error_comparison.png")
    
    println("\nPlots saved as 'runtime_comparison.png' and 'error_comparison.png'")
    println("\nAll tests completed successfully!")
end

main() 