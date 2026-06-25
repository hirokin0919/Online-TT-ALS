"""
Reconstruct a single data slice from the spatial TT cores and the weight vector.

This function sequentially contracts the spatial TT cores {G_k}_{k=1}^{N-1} 
and the time-specific weight vector `g_n` using efficient matrix multiplications.

Inputs:
- G: Array of spatial TT cores.
- g: The time-specific weight vector at the final mode.

Returns:
- X_slice: The reconstructed original data slice.
"""
function ttProduct(G::Vector{Array{Float64}}, g::Vector{Float64})
    N = length(G) + 1
    P = G[1]

    for n in 2:N-1
        r1, i, r2 = size(G[n])
        P = reshape(P, :, r1)
        Gn = reshape(G[n], r1, :)
        P = P * Gn
    end
    P = reshape(P, :, size(G[N-1], 3))
    X_vec = P * g
    X_slice = reshape(X_vec, map(x -> size(x, 2), G)...)

    return X_slice
end     

"""
Reconstruct the full multi-dimensional tensor from a given set of TT cores.

Inputs:
- G: Array of TT cores.

Returns:
- full_tensor: The reconstructed dense tensor.
"""
function tt2full(G::Vector{Array{Float64}})
    N = length(G)
    full_tensor = G[1]

    for n in 2:N
        r1, i, r2 = size(G[n])
        full_tensor = reshape(full_tensor, :, size(full_tensor, ndims(full_tensor)))
        Gn_reshaped = reshape(G[n], r1, i * r2)
        full_tensor = full_tensor * Gn_reshaped
        full_tensor = reshape(full_tensor, :, i, r2)
    end

    if size(G[end])[end] == 1
        return reshape(full_tensor, map(x -> size(x, 2), G)...)
    else
        return reshape(full_tensor, map(x -> size(x, 2), G)..., size(G[end])[end])
    end
end