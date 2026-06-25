"""
Implementation of the proposed Online TT-ALS algorithm.

This function represents the core contribution of our framework: an exact, single-sweep 
online Tensor Train decomposition for streaming data. By strictly enforcing orthogonal 
gauge constraints dynamically, it bypasses the quadratic rank dependency found in 
existing first-order approximations and ensures monotonic error convergence.

Inputs:
- X_stream: Streaming data tensor of size (I_1, ..., I_n-1, T).
- ttsizes: Vector of spatial dimensions and time steps.
- ttranks: Target TT-ranks (length N-1).

Returns:
- G_history: History of the spatial TT cores at each time step.
- gn_history: History of the time-varying weight vectors.
- iter_time: Processing time per frame (in milliseconds).
"""
function online_ttALS(X_stream::Array{Float64}, ttsizes::Vector{Int}, ttranks::Vector{Int})
    N = length(ttsizes)
    T = ttsizes[end]
    r = [1; ttranks; 1]

    # Initialize spatial TT cores with random orthogonal values
    G = Vector{Array{Float64}}(undef, N-1)
    for n in 1:N-1
        G[n] = randn(r[n], ttsizes[n], r[n+1])
        G[n] /= norm(G[n])
    end

    # Initialize the time-varying weight vector
    g_n = randn(r[N])
    g_n /= norm(g_n)

    G_history = Vector{Vector{Array{Float64}}}(undef, T)
    gn_history = Vector{Vector{Float64}}(undef, T)
    iter_time = zeros(T)

    # Pre-allocate buffers for the right contraction matrices (B)
    B_list_buffers = Vector{Matrix{Float64}}(undef, N-1)
    B_list_buffers[N-1] = Matrix{Float64}(undef, r[N], 1)
    curr_right_dim = 1
    for k in (N-2):-1:1
        curr_right_dim *= ttsizes[k+1]
        B_list_buffers[k] = Matrix{Float64}(undef, r[k+1], curr_right_dim)
    end

    # Pre-allocate buffers for the left contraction matrices (A)
    A_buffers = Vector{Matrix{Float64}}(undef, N)
    A_buffers[1] = Matrix{Float64}(undef, 1, 1)
    curr_left_dim = 1
    for k in 1:N-1
        curr_left_dim *= ttsizes[k]
        A_buffers[k+1] = Matrix{Float64}(undef, curr_left_dim, r[k+1])
    end

    # === Main Streaming Loop ===
    for t in 1:T
        t0 = time_ns()

        # Extract the current data slice
        X_slice = @view X_stream[ntuple(_ -> :, N-1)..., t]

        # Normalize the weight vector to prevent numerical overflow
        nrm = norm(g_n)
        if nrm > 1e-12
            g_n /= nrm
        end
        G[N-1] .*= nrm

        # 1. Right-Orthogonalization Phase
        for k in (N-1):-1:2
            rk1, ik, rk2 = size(G[k])

            # Reshape into matrix: (r_prev) x (Ik * r_curr)
            Gk_mat = reshape(G[k], rk1, :) 

            # Perform LQ decomposition (L: lower triangular, Q: orthogonal)
            F = lq!(Gk_mat)
            L = Matrix(F.L)
            Q = Matrix(F.Q)

            # Update the core with the orthogonalized factor Q
            G[k] = reshape(Q, rk1, ik, rk2)
            
            # Absorb L into the left adjacent core
            G_prev_mat = reshape(G[k-1], :, rk1)
            G[k-1] = reshape(G_prev_mat * L, size(G[k-1])...)
        end

        # 2. Precompute the right contraction matrices (B)
        copyto!(B_list_buffers[N-1], g_n)
        
        for k in (N-2):-1:1
            rk1, ik, rk2 = size(G[k+1])
            # G[k+1]: (r[k+1], I_{k+1}, r[k+2])
            # B_next: (r[k+2], :)
            G_next_flat = reshape(G[k+1], :, rk2)
            B_next_flat = B_list_buffers[k+1]
            
            target_view = reshape(B_list_buffers[k], rk1*ik, :)
            mul!(target_view, G_next_flat, B_next_flat) # (r[k+1]*I_{k+1}, :)
        end

        # 3. Initialize the left contraction matrix (A)
        A = A_buffers[1]
        fill!(A, 1.0)

        # 4. Forward Update and Left-Orthogonalization Phase
        for k in 1:N-1
            # X: (Left, I_k, Right)
            left_dim = prod(ttsizes[1:k-1])
            mid_dim = ttsizes[k]
            right_dim = prod(ttsizes[k+1:N-1])
            
            X_mat = reshape(X_slice, left_dim, mid_dim * right_dim)
            
            # Left contraction: A^T * X -> (r[k], Mid * Right)
            AX = A' * X_mat 
            
            # Right contraction: (AX) * B^T -> (r[k] * Mid, r[k+1])
            AX_reshaped = reshape(AX, r[k] * mid_dim, right_dim)

            # Y_flat = AX * B' -> (r[k]*Mid, r[k+1])
            Y_flat = AX_reshaped * B_list_buffers[k]' 
            
            # Update the current core based on the local exact solution
            G[k] = reshape(Y_flat, r[k], mid_dim, r[k+1])
            
            # Left-Orthogonalization via QR decomposition
            Gk_mat = reshape(G[k], r[k]*mid_dim, r[k+1])
            F = qr!(Gk_mat)
            Q = Matrix(F.Q)
            R = Matrix(F.R)
            
            # Update G[k] with the orthogonalized factor
            G[k] = reshape(Q, r[k], mid_dim, r[k+1])
            
            # Propagate the residual R to the next stage
            if k < N-1
                # Absorb R into the left interface of G[k+1]
                r_nxt, I_nxt, r_nxt2 = size(G[k+1])
                G_next_mat = reshape(G[k+1], r_nxt, :)
                G[k+1] = reshape(R * G_next_mat, r_nxt, I_nxt, r_nxt2)
            else
                # For the last core, absorb R into the time-varying weight vector
                g_n = R * g_n
            end
            
            # Update the left interface matrix A for the next step
            Gk_flat = reshape(G[k], r[k], mid_dim * r[k+1])
            target_reshaped = reshape(A_buffers[k+1], left_dim, mid_dim * r[k+1])
            mul!(target_reshaped, A, Gk_flat) # (Left, I_k * r[k+1])
            A = A_buffers[k+1]
        end

        # 5. Update the time-varying weight vector g_n
        X_vec = reshape(X_slice, :)
        
        # # Calculate g_n = A^T * X
        # A: (Left_All, r[N])
        # X: (Left_All, 1)
        mul!(g_n, A', X_vec) # (r[N], 1)

        t1 = time_ns()
        iter_time[t] = (t1 - t0) / 1e6 # Convert to milliseconds

        # Save state for historical tracking
        G_history[t] = deepcopy(G)
        gn_history[t] = deepcopy(g_n)
    end

    return G_history, gn_history, iter_time
end

"""
Implementation of the Online Tensor Train First-Order Approximation (TT-FOA) algorithm.

Note: 
The theoretical algorithm implemented in this function was originally proposed by 
Thanh et al. [1]. This code represents our independent implementation of their 
methodology to serve as a baseline for comparative evaluation.

[1] L. T. Thanh, K. Abed-Meraim, N. L. Trung, and R. Boyer. "Adaptive Algorithms 
    for Tracking Tensor-Train Decomposition of Streaming Tensors." In 2020 28th 
    European Signal Processing Conference (EUSIPCO), pp. 995-999, 2021.

Inputs:
- X_stream: Streaming data tensor of size (I_1, ..., I_n-1, T).
- ttsizes: Spatial dimensions and time steps.
- ttranks: Vector of TT-ranks.
- lambda: Forgetting factor for the Recursive Least Squares (RLS).
- rho: Regularization parameter.
- sketchsize: Number of samples for sketching (0 means no sketching).

Returns:
- G_history: History of the spatial TT cores at each time step.
- gn_history: History of the time-varying weight vectors.
- iter_time: Processing time per frame (in milliseconds).
"""
function ttFOA(X_stream::Array{Float64}, ttsizes::Vector{Int}, ttranks::Vector{Int}, lambda::Float64=0.7, rho::Float64=0.0, sketchsize::Int=0)
    N = length(ttsizes)
    T = ttsizes[end]
    r = [1; ttranks; 1]

    # 1. Initialization
    G = Vector{Array{Float64}}(undef, N-1)
    S = Vector{Matrix{Float64}}(undef, N-1)
    for n in 1:N-1
        G[n] = randn(r[n], ttsizes[n], r[n+1])
        # Initialize the covariance matrices for RLS
        S[n] = Matrix{Float64}(I, r[n]*r[n+1], r[n]*r[n+1])*100.0
    end

    G_history = Vector{Vector{Array{Float64}}}(undef, T)
    gn_history = Vector{Vector{Float64}}(undef, T)
    iter_time = zeros(T)

    for t in 1:T
        t0 = time_ns()
        X_slice = X_stream[ntuple(_ -> :, N-1)..., t]
        x_vec = reshape(X_slice, :)

        # Compute the contraction of all spatial cores
        H_tm1 = G[1]
        for k in 2:N-1
            r1, i, r2 = size(G[k])
            H_tm1 = reshape(H_tm1, :, r1)
            Gk_reshaped = reshape(G[k], r1, :)
            H_tm1 = H_tm1 * Gk_reshaped
        end
        H_tm1_mat = reshape(H_tm1, :, r[end-1])
        
        # Apply sketching if specified to reduce computational complexity
        if sketchsize == 0
            H_sketch = H_tm1_mat
            x_sketch = x_vec
        else
            sketch_indices = rand(1:size(H_tm1_mat, 1), min(sketchsize, size(H_tm1_mat, 1)))
            H_sketch = H_tm1_mat[sketch_indices, :]
            x_sketch = x_vec[sketch_indices]
        end

        # Update the time-varying weight vector via Ridge Regression
        g_n = (H_sketch' * H_sketch + rho * I) \ (H_sketch' * x_sketch)
        gn_history[t] = g_n

        # Compute the residual tensor
        Delta_t = X_slice - reshape(H_tm1_mat * g_n, ttsizes[1:end-1]...)

        A = Vector{Matrix{Float64}}(undef, N-1)
        B = Vector{Matrix{Float64}}(undef, N-1)
    
        # Compute left interface matrices A
        A[1] = ones(1, 1)
        A[2] = reshape(G[1], :, r[2])
        for k in 3:N-1
            A[k] = reshape(A[k-1]*reshape(G[k-1], r[k-1], :), :, r[k])
        end
        

        # Compute right interface matrices B
        B[N-1] = reshape(g_n, r[N], 1)
        B[N-2] = reshape(reshape(G[N-1], :, r[N])*g_n, r[N-1], :)
        for k in N-3:-1:1
            B[k] = reshape(reshape(G[k+1], :, r[k+2])*B[k+1], r[k+1], :)
        end

        # Update each spatial core sequentially based on first-order approximation
        for k in 1:N-1
            Gk_old_mat = reshape(permutedims(G[k], (2,1,3)), ttsizes[k], :)
            A_k = A[k]'
            B_k = B[k]

            # Construct the joint interface matrix and update the RLS covariance
            W_k = kron(B_k, A_k)
            S[k] = lambda*S[k] + W_k*W_k'

            # Compute the update step
            V_k = W_k'*pinv(S[k])
            Delta_k = reshape(permutedims(Delta_t, (k, setdiff(1:N-1, k)...)), ttsizes[k], :)
            Gk_new_mat = Gk_old_mat + Delta_k*V_k

            # Restore the updated core to its original tensor shape
            G[k] = permutedims(reshape(Gk_new_mat, ttsizes[k], r[k], r[k+1]), (2,1,3))
        end

        t1 = time_ns()
        iter_time[t] = (t1 - t0) / 1e6 # Convert to milliseconds
        G_history[t] = deepcopy(G)
        gn_history[t] = deepcopy(gn_history[t])
    end

    return G_history, gn_history, iter_time
end

"""
Implementation of the batch Alternating Least Squares (TT-ALS) algorithm.

Note:
The Alternating Linear Scheme for TT optimization was originally proposed by 
Holtz et al. [1]. This code represents our optimized implementation of their 
batch algorithm, augmented with right interface caching for faster execution.

[1] S. Holtz, T. Rohwedder, and R. Schneider. "The Alternating Linear Scheme 
    for Tensor Optimization in the Tensor Train Format." SIAM Journal on 
    Scientific Computing, 34(2):A683-A713, 2012.

Inputs:
- X_full: The complete dense tensor to be approximated.
- ttranks: Target TT-ranks (length N-1).
- max_sweeps: Maximum number of bidirectional ALS sweeps.
- tol: Tolerance for the relative reconstruction error to trigger early stopping.

Returns:
- G: Array of optimized TT cores.
"""
function batch_ttALS(X_full::Array{Float64}, ttranks::Vector{Int}; max_sweeps::Int=10, tol::Float64=1e-6)
    ttsizes = collect(size(X_full))
    N = length(ttsizes)
    r = [1; ttranks; 1]

    # 1. Initialization: Generate random orthogonal TT cores
    G = Vector{Array{Float64}}(undef, N)
    for n in 1:N
        G[n] = randn(r[n], ttsizes[n], r[n+1])
        G[n] /= norm(G[n])
    end

    # 2. Right-to-Left Orthogonalization
    # Ensure that the initial cores satisfy the right-orthogonality constraints
    for n in N:-1:2
        G_mat = reshape(G[n], r[n], :) # (r[n], I_n * r[n+1])
        F = lq(G_mat)
        L = Matrix(F.L)
        Q = Matrix(F.Q)
        
        G[n] = reshape(Q, r[n], ttsizes[n], r[n+1])
        
        # Absorb the lower triangular matrix L into the left adjacent core
        G_prev = reshape(G[n-1], :, r[n])
        G[n-1] = reshape(G_prev * L, size(G[n-1])...)
    end

    # 3. Precompute the Right Interface matrices
    # Right_Inter[k] caches the contraction of cores G[k]...G[N]
    # This prevents redundant calculations during the optimization sweeps
    Right_Inter = Vector{Matrix{Float64}}(undef, N+1)
    Right_Inter[N+1] = ones(1, 1) # Dummy boundary condition
    
    # Accumulate from right to left
    for n in N:-1:2
        G_n = reshape(G[n], r[n], :) # (r[n], I_n * r[n+1])
        R_next = Right_Inter[n+1]
        
        # Matrix multiplication representing the tensor contraction
        sz_n = size(G[n]) # (r[n], I_n, r[n+1])
        G_temp = reshape(G[n], sz_n[1] * sz_n[2], sz_n[3])
        R_curr = G_temp * R_next # (r[n]*I_n, Right_Dim)
        
        # Reshape to (r[n], Rest) to facilitate the ALS update equations
        Right_Inter[n] = reshape(R_curr, sz_n[1], :)
    end

    last_error = Inf

    # === Main ALS Loop ===
    for sweep in 1:max_sweeps
        # --- Forward Sweep (Left -> Right) ---
        # The left interface matrices are accumulated dynamically within the loop
        Left_Inter = ones(1, 1) # Initial state (r[1], Left_Dim=1)

        for n in 1:N-1
            # 1. Optimize the local core G[n]
            # Formulate the local subproblem: min || X - L * G[n] * R ||
            # Solution: G[n] = L^T * X * R^T (Due to orthogonal constraints)
            
            dim_left = prod(ttsizes[1:n-1])
            dim_mid = ttsizes[n]
            dim_right = prod(ttsizes[n+1:N])
            
            # Step 1: Left contraction P = L^T * X
            X_reshaped = reshape(X_full, dim_left, dim_mid * dim_right)
            P = Left_Inter' * X_reshaped # -> (r[n], Mid * Right)
            
            # Step 2: Right contraction Y = P * R^T
            P = reshape(P, r[n] * dim_mid, dim_right)
            R_mat = Right_Inter[n+1] 
            
            Y = P * R_mat' # -> (r[n] * Mid, r[n+1])
            
            # This yields the newly updated G[n]
            G_new = reshape(Y, r[n], dim_mid, r[n+1])
            
            # 2. Left-Orthogonalization via QR Decomposition
            G_mat = reshape(G_new, r[n] * dim_mid, r[n+1])
            F_qr = qr(G_mat)
            Q = Matrix(F_qr.Q)
            R = Matrix(F_qr.R)
            
            # Update G[n] with the orthogonalized factor Q
            G[n] = reshape(Q, r[n], dim_mid, r[n+1])
            
            # Absorb the residual R into the right adjacent core G[n+1]
            sz_next = size(G[n+1])
            G_next_flat = reshape(G[n+1], sz_next[1], :)
            G[n+1] = reshape(R * G_next_flat, sz_next...)
            
            # 3. Update the Left Interface for the next iteration
            G_flat = reshape(G[n], r[n], :)
            Left_Inter_Next = Left_Inter * G_flat # (Left_Dim, I_n * r[n+1])
            Left_Inter = reshape(Left_Inter_Next, dim_left * dim_mid, r[n+1])
        end
        
        # --- Backward Sweep (Right -> Left) ---
        # Reset the right interface matrix for dynamic accumulation
        Curr_Right_Inter = ones(1, 1) # (r[N+1], Right_Dim=1)

        for n in N:-1:2
            # 1. Optimize the local core G[n]
            dim_left = prod(ttsizes[1:n-1])
            dim_mid = ttsizes[n]
            dim_right = prod(ttsizes[n+1:N])
            
            X_reshaped = reshape(X_full, dim_left * dim_mid, dim_right)
            
            # Right contraction: P = X * R^T
            P = X_reshaped * Curr_Right_Inter' # -> (Left * Mid, r[n+1])
            P = reshape(P, dim_left, dim_mid * r[n+1])
            
            # Left contraction: Y = L^T * P
            # Note: To optimize memory, we recompute the left contraction on the fly 
            # rather than storing all intermediate left interfaces during the forward sweep.
            L_mat = compute_left_contraction(G, 1, n-1)
            
            Y = L_mat' * P # (r[n], Mid * r[n+1])
            
            G_new = reshape(Y, r[n], dim_mid, r[n+1])
            
            # 2. Right-Orthogonalization via LQ Decomposition
            G_mat = reshape(G_new, r[n], :)
            F_lq = lq(G_mat)
            L = Matrix(F_lq.L)
            Q = Matrix(F_lq.Q)
            
            G[n] = reshape(Q, r[n], dim_mid, r[n+1])
            
            # Absorb the lower triangular matrix L into the left adjacent core
            sz_prev = size(G[n-1])
            G_prev_flat = reshape(G[n-1], :, sz_prev[3])
            G[n-1] = reshape(G_prev_flat * L, sz_prev...)
            
            # 3. Update the Right Interface for the next iteration
            G_flat = reshape(G[n], r[n] * dim_mid, r[n+1])
            Next_R = G_flat * Curr_Right_Inter # (r[n]*I_n, Right_Dim)
            
            Curr_Right_Inter = reshape(Next_R, r[n], :)
        end

        # Evaluate convergence based on the relative reconstruction error
        X_est = tt2full_batch(G)
        curr_error = norm(X_full - X_est) / norm(X_full)
        
        if abs(last_error - curr_error) < tol
            break
        end
        last_error = curr_error
    end

    return G
end

# --- Helper Functions ---

"""
Compute the left contraction matrix for cores G[idx_start] to G[idx_end].
Used dynamically during the backward sweep of batch_ttALS to save memory.
"""
function compute_left_contraction(G, idx_start, idx_end)
    # Init: G[start] -> (r[start]*I_start, r[start+1])
    res = G[idx_start]
    r1, i, r2 = size(res)
    res = reshape(res, r1 * i, r2)
    
    for k in idx_start+1:idx_end
        rk, ik, rk1 = size(G[k])
        G_flat = reshape(G[k], rk, ik * rk1)
        res = res * G_flat # (Left_Dim, I_k * r[k+1])
        res = reshape(res, :, rk1)
    end
    return res
end

"""
Reconstruct the full dense tensor from an array of TT cores (batch version).
"""
function tt2full_batch(G::Vector{Array{Float64}})
    N = length(G)
    full_tensor = G[1]
    for n in 2:N
        r1, i, r2 = size(G[n])
        full_tensor = reshape(full_tensor, :, size(full_tensor, ndims(full_tensor)))
        Gn_reshaped = reshape(G[n], r1, i * r2)
        full_tensor = full_tensor * Gn_reshaped
        full_tensor = reshape(full_tensor, :, i, r2)
    end
    return reshape(full_tensor, map(x -> size(x, 2), G)...)
end