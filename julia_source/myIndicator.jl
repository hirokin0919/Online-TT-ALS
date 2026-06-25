"""
Calculate the Peak Signal-to-Noise Ratio (PSNR) for a grayscale video.

Inputs:
- X_true: Ground-truth video tensor of size (H, W, T) with values in [0.0, 1.0].
- X_est: Reconstructed video tensor of size (H, W, T).

Returns:
- avg_psnr: The mean PSNR across all frames.
- psnr_list: Frame-by-frame PSNR values.
"""
function calc_video_psnr_gray(X_true, X_est)
    T = size(X_true, 3)
    psnr_list = Float64[]
    
    for t in 1:T
        # 1. Extract and clamp data
        # ImageQualityIndexes expects values in [0.0, 1.0]. Low-rank approximations 
        # may occasionally yield slight overshoots or negative values.
        slice_true = clamp.(X_true[:, :, t], 0.0, 1.0)
        slice_est = clamp.(X_est[:, :, t], 0.0, 1.0)
        
        # 2. Convert numerical arrays to Gray image objects
        img_true = Gray.(slice_true)
        img_est = Gray.(slice_est)
        
        # 3. Calculate PSNR
        # For Gray type inputs, maxval is automatically assumed to be 1.0.
        score = assess_psnr(img_true, img_est)
        
        push!(psnr_list, score)
    end
    
    # Calculate the average PSNR over all frames
    avg_psnr = mean(psnr_list)
    
    return avg_psnr, psnr_list
end

"""
Calculate the Peak Signal-to-Noise Ratio (PSNR) for a color video.

Inputs:
- X_true: Ground-truth video tensor of size (H, W, 3, T) with values in [0.0, 1.0].
- X_est: Reconstructed video tensor of size (H, W, 3, T).
"""
function calc_video_psnr_color(X_true, X_est)
    T = size(X_true, 4)
    psnr_list = Float64[]

    for t in 1:T
        # 1. Extract and clamp data
        slice_true = clamp.(X_true[:, :, :, t], 0.0, 1.0)
        slice_est = clamp.(X_est[:, :, :, t], 0.0, 1.0)

        # 2. Convert to RGB image objects
        # Julia's Images package expects the layout (Channels, H, W).
        # We permute the dimensions from (H, W, 3) to (3, H, W) before coloring.
        img_true = colorview(RGB, permutedims(slice_true, (3, 1, 2)))
        img_est = colorview(RGB, permutedims(slice_est, (3, 1, 2)))

        # 3. Calculate PSNR for RGB channels
        score = assess_psnr(img_true, img_est)
        push!(psnr_list, score)
    end

    return mean(psnr_list), psnr_list
end

"""
Calculate the Masked Root Mean Square Error (M-RMSE) for a grayscale video.
This metric focuses exclusively on the moving objects (foreground) by applying background subtraction.

Inputs:
- X_true: Ground-truth tensor (H, W, T).
- X_est: Reconstructed tensor (H, W, T).
- threshold: The intensity threshold to separate foreground from background.

Returns:
- Average M-RMSE across all frames.
"""
function calc_masked_rmse_gray(X_true, X_est; threshold=0.1)
    # 1. Estimate the true background image by computing the temporal median
    # dropdims removes the singleton time dimension, resulting in (H, W)
    background_img = dropdims(median(X_true, dims=3), dims=3)
    
    masked_rmse_list = Float64[]
    T = size(X_true, 3)
    
    for t in 1:T
        frame_true = X_true[:, :, t]
        frame_est = X_est[:, :, t]
        
        # 2. Create a binary mask for the moving foreground object
        # Pixels with a large absolute difference from the background are classified as foreground
        mask = abs.(frame_true .- background_img) .> threshold
        
        # Count the number of foreground pixels (area of the moving object)
        num_pixels = sum(mask)
        
        if num_pixels > 0
            # 3. Calculate error exclusively within the masked region
            diff = (frame_true .- frame_est) .* mask
            
            # RMSE: sqrt( Sum of Squared Errors / Number of Masked Pixels )
            mse = sum(diff .^ 2) / num_pixels
            rmse = sqrt(mse)
            
            push!(masked_rmse_list, rmse)
        else
            # If no moving object is detected, append 0.0 to avoid NaN
            push!(masked_rmse_list, 0.0)
        end
    end
    
    return mean(masked_rmse_list)
end

"""
Calculate the Masked Root Mean Square Error (M-RMSE) for a color video.
"""
function calc_masked_rmse_color(X_true, X_est; threshold=0.1)
    # 1. Estimate the true background image by computing the temporal median
    # dropdims removes the singleton time dimension, resulting in (H, W, 3)
    background_img = dropdims(median(X_true, dims=4), dims=4)
    
    masked_rmse_list = Float64[]
    
    T = size(X_true, 4)
    
    for t in 1:T
        frame_true = X_true[:, :, :, t]
        frame_est = X_est[:, :, :, t]
        
        # 2. Create a binary mask for the moving foreground object
        mask = abs.(frame_true .- background_img) .> threshold

        num_pixels = sum(mask)
        
        if num_pixels > 0
            # 3. Calculate error exclusively within the masked region
            diff = (frame_true .- frame_est) .* mask
            
            mse = sum(diff .^ 2) / num_pixels
            rmse = sqrt(mse)
            
            push!(masked_rmse_list, rmse)
        else
            push!(masked_rmse_list, 0.0)
        end
    end
    
    return mean(masked_rmse_list)
end

"""
Calculate Structural Similarity (SSIM) for a grayscale video.
"""
function calc_video_ssim_gray(X_true, X_est)
    T = size(X_true, 3)
    ssim_list = Float64[]
    
    for t in 1:T
        slice_true = X_true[:, :, t]
        slice_est = X_est[:, :, t]
        
        # Safety measure: SSIM is sensitive to values outside the [0, 1] range.
        slice_true = clamp.(slice_true, 0.0, 1.0)
        slice_est = clamp.(slice_est, 0.0, 1.0)
        
        img_true = Gray.(slice_true)
        img_est = Gray.(slice_est)
        
        score = assess_ssim(img_true, img_est)
        push!(ssim_list, score)
    end
    
    return mean(ssim_list), ssim_list
end

"""
Calculate Structural Similarity (SSIM) for a color video.
"""
function calc_video_ssim_color(X_true, X_est)
    T = size(X_true, 4)
    ssim_list = Float64[]

    for t in 1:T
        slice_true = clamp.(X_true[:, :, :, t], 0.0, 1.0)
        slice_est = clamp.(X_est[:, :, :, t], 0.0, 1.0)

        img_true = colorview(RGB, permutedims(slice_true, (3, 1, 2)))
        img_est = colorview(RGB, permutedims(slice_est, (3, 1, 2)))

        # assess_ssim automatically handles RGB components
        score = assess_ssim(img_true, img_est)
        push!(ssim_list, score)
    end

    return mean(ssim_list), ssim_list
end

"""
Calculate Multi-Scale Structural Similarity (MS-SSIM) for a grayscale video.
"""
function calc_video_msssim_gray(X_true, X_est)
    T = size(X_true, 3)
    msssim_list = Float64[]
    
    for t in 1:T
        slice_true = clamp.(X_true[:, :, t], 0.0, 1.0)
        slice_est = clamp.(X_est[:, :, t], 0.0, 1.0)
        
        img_true = Gray.(slice_true)
        img_est = Gray.(slice_est)
        
        # MS-SSIM requires a sufficiently large image resolution.
        # We wrap it in a try-catch block to handle frames that are too small for the default 5-scale pyramid.
        try
            score = assess_msssim(img_true, img_est)
            push!(msssim_list, score)
        catch e
            println("Frame $t: MS-SSIM calculation failed (likely due to small image size).")
            push!(msssim_list, NaN) 
        end
    end
    
    # Compute the average excluding any NaN values
    valid_scores = filter(!isnan, msssim_list)
    avg_score = isempty(valid_scores) ? 0.0 : mean(valid_scores)
    
    return avg_score, msssim_list
end

"""
Calculate Multi-Scale Structural Similarity (MS-SSIM) for a color video.
"""
function calc_video_msssim_color(X_true, X_est)
    T = size(X_true, 4)
    msssim_list = Float64[]

    for t in 1:T
        slice_true = clamp.(X_true[:, :, :, t], 0.0, 1.0)
        slice_est = clamp.(X_est[:, :, :, t], 0.0, 1.0)

        img_true = colorview(RGB, permutedims(slice_true, (3, 1, 2)))
        img_est = colorview(RGB, permutedims(slice_est, (3, 1, 2)))

        try
            score = assess_msssim(img_true, img_est)
            push!(msssim_list, score)
        catch e
            push!(msssim_list, NaN)
        end
    end

    valid_scores = filter(!isnan, msssim_list)
    avg_score = isempty(valid_scores) ? 0.0 : mean(valid_scores)

    return avg_score, msssim_list
end

"""
Calculate Learned Perceptual Image Patch Similarity (LPIPS) for a grayscale video.

Note: This function assumes `torch` and a pre-initialized `loss_fn` (e.g., lpips.LPIPS) 
are available in the current PyCall/PythonCall context.
"""
function calc_video_lpips_gray(X_true, X_est)
    T = size(X_true, 3)
    lpips_list = Float64[]
    
    # LPIPS expects input values normalized to the range [-1.0, 1.0]
    normalize(x) = (clamp(x, 0.0, 1.0) * 2.0) - 1.0
    
    for t in 1:T
        slice_true = X_true[:, :, t]
        slice_est = X_est[:, :, t]
        
        # 1. Duplicate grayscale to 3 channels (RGB) to match the expected CNN input
        # (H, W) -> (H, W, 3)
        img_true_rgb = cat(slice_true, slice_true, slice_true, dims=3)
        img_est_rgb = cat(slice_est, slice_est, slice_est, dims=3)
        
        # 2. Permute dimensions: Julia (H, W, C) -> PyTorch (C, H, W)
        img_true_chw = permutedims(img_true_rgb, (3, 1, 2))
        img_est_chw = permutedims(img_est_rgb, (3, 1, 2))
        
        # 3. Convert to PyTorch tensor, normalize, and add the batch dimension (1, C, H, W)
        tensor_true = torch.tensor(normalize.(img_true_chw)).float().unsqueeze(0)
        tensor_est = torch.tensor(normalize.(img_est_chw)).float().unsqueeze(0)
        
        # 4. Evaluate LPIPS distance and extract the scalar value
        dist = loss_fn(tensor_true, tensor_est).item()
        
        push!(lpips_list, dist)
    end
    
    return mean(lpips_list), lpips_list
end

"""
Calculate Learned Perceptual Image Patch Similarity (LPIPS) for a color video.
"""
function calc_video_lpips_color(X_true, X_est)
    T = size(X_true, 4)
    lpips_list = Float64[]
    
    normalize(x) = (clamp(x, 0.0, 1.0) * 2.0) - 1.0

    for t in 1:T
        slice_true = X_true[:, :, :, t]
        slice_est = X_est[:, :, :, t]

        # Permute dimensions: Julia (H, W, C) -> PyTorch (C, H, W)
        img_true_chw = permutedims(slice_true, (3, 1, 2))
        img_est_chw = permutedims(slice_est, (3, 1, 2))

        # Add batch dimension: (1, 3, H, W)
        tensor_true = torch.tensor(normalize.(img_true_chw)).float().unsqueeze(0)
        tensor_est = torch.tensor(normalize.(img_est_chw)).float().unsqueeze(0)

        dist = loss_fn(tensor_true, tensor_est).item()
        push!(lpips_list, dist)
    end

    return mean(lpips_list), lpips_list
end

"""
Helper function to export uncompressed grayscale video tensors to the .y4m (YUV4MPEG2) format.
This format is required for standard VMAF evaluation via FFmpeg.
"""
function write_y4m_gray(tensor, filename; fps=30)
    # If the input is (H, W, T), replicate it across 3 channels to simulate RGB/YUV structure
    if ndims(tensor) == 3
        H, W, T = size(tensor)
        tensor_4d = reshape(tensor, H, W, 1, T)
        tensor = cat(tensor_4d, tensor_4d, tensor_4d, dims=3)
    end
    
    H, W, C, T = size(tensor)
    
    open(filename, "w") do io
        # 1. Write the Y4M header (assuming YUV444p, 8-bit)
        println(io, "YUV4MPEG2 W$W H$H F$fps:1 Ip A1:1 C444")
        
        for t in 1:T
            println(io, "FRAME")
            
            # Extract frame and scale to 8-bit integer [0, 255]
            frame = tensor[:, :, :, t]
            frame_u8 = round.(UInt8, clamp.(frame, 0.0, 1.0) .* 255)
            
            # 2. RGB to YUV BT.709 conversion
            # VMAF requires the YUV color space. We convert the RGB matrices accordingly.
            R = Float64.(frame_u8[:, :, 1])
            G = Float64.(frame_u8[:, :, 2])
            B = Float64.(frame_u8[:, :, 3])
            
            Y =  0.2126 * R + 0.7152 * G + 0.0722 * B
            U = -0.1146 * R - 0.3854 * G + 0.5000 * B .+ 128
            V =  0.5000 * R - 0.4542 * G - 0.0458 * B .+ 128
            
            Y_bytes = round.(UInt8, clamp.(Y, 0, 255))
            U_bytes = round.(UInt8, clamp.(U, 0, 255))
            V_bytes = round.(UInt8, clamp.(V, 0, 255))
            
            # Write raw binary data (planar format: all Y, then all U, then all V)
            write(io, Y_bytes)
            write(io, U_bytes)
            write(io, V_bytes)
        end
    end
end

"""
Helper function to export uncompressed color video tensors to the .y4m (YUV4MPEG2) format.
"""
function write_y4m_color(tensor, filename; fps=30)
    H, W, C, T = size(tensor)
    
    open(filename, "w") do io
        # Write the Y4M header specifying C444 (no chroma subsampling)
        println(io, "YUV4MPEG2 W$W H$H F$fps:1 Ip A1:1 C444")
        
        for t in 1:T
            println(io, "FRAME")
            
            frame = tensor[:, :, :, t]
            frame_u8 = round.(UInt8, clamp.(frame, 0.0, 1.0) .* 255)

            R = Float64.(frame_u8[:, :, 1])
            G = Float64.(frame_u8[:, :, 2])
            B = Float64.(frame_u8[:, :, 3])
            
            # RGB to YUV BT.709 conversion
            Y =  0.2126 * R + 0.7152 * G + 0.0722 * B
            U = -0.1146 * R - 0.3854 * G + 0.5000 * B .+ 128
            V =  0.5000 * R - 0.4542 * G - 0.0458 * B .+ 128
            
            Y_bytes = round.(UInt8, clamp.(Y, 0, 255))
            U_bytes = round.(UInt8, clamp.(U, 0, 255))
            V_bytes = round.(UInt8, clamp.(V, 0, 255))
            
            write(io, Y_bytes)
            write(io, U_bytes)
            write(io, V_bytes)
        end
    end
end

"""
Evaluate Video Multi-method Assessment Fusion (VMAF) for a grayscale video via FFmpeg.

Inputs:
- X_true: Ground-truth video tensor.
- X_est: Reconstructed video tensor.
- model: The VMAF model version to use.

Returns:
- The overall VMAF score [0, 100].
"""
function calc_video_vmaf_gray(X_true, X_est; model="vmaf_v0.6.1")
    # NOTE: Users cloning this repository should update this path to their local FFmpeg executable.
    ffmpeg_path = "/home/takeda/ffmpeg_tool/ffmpeg-7.0.2-amd64-static/ffmpeg"

    # Generate temporary files for the reference, distorted video, and log
    ref_file = tempname() * ".y4m"
    dist_file = tempname() * ".y4m"
    log_file = tempname() * ".json"
    
    try
        # 1. Export tensors to uncompressed .y4m files
        write_y4m_gray(X_true, ref_file)
        write_y4m_gray(X_est, dist_file)
        
        # 2. Construct and execute the FFmpeg command
        # libvmaf evaluates the distorted file against the reference file.
        cmd = `$ffmpeg_path -hide_banner -v error -r 30 -i $dist_file -r 30 -i $ref_file -lavfi "libvmaf=model=version=$model:log_path=$log_file:log_fmt=json" -f null -`
        run(cmd)
        
        # 3. Parse the overall VMAF score from the JSON log output
        json_content = read(log_file, String)
        
        # Use regex to extract the aggregated mean VMAF score located at the end of the JSON log
        matches = collect(eachmatch(r"\"mean\":\s*([\d\.]+)", json_content))
        if !isempty(matches)
            vmaf_score = parse(Float64, matches[end].captures[1])
            return vmaf_score
        else
            println("Error: Could not parse VMAF score from output.")
            return NaN
        end
        
    catch e
        println("VMAF Calculation Failed: $e")
        return NaN
        
    finally
        # Clean up temporary files
        rm(ref_file, force=true)
        rm(dist_file, force=true)
        rm(log_file, force=true)
    end
end



"""
Evaluate Video Multi-method Assessment Fusion (VMAF) for a color video via FFmpeg.
"""
function calc_video_vmaf_color(X_true, X_est; model="vmaf_v0.6.1")
    # NOTE: Users cloning this repository should update this path to their local FFmpeg executable.
    ffmpeg_path = "/home/takeda/ffmpeg_tool/ffmpeg-7.0.2-amd64-static/ffmpeg"

    ref_file = tempname() * ".y4m"
    dist_file = tempname() * ".y4m"
    log_file = tempname() * ".json"

    try
        # 1. Export tensors to uncompressed .y4m files
        write_y4m_color(X_true, ref_file)
        write_y4m_color(X_est, dist_file)

        # 2. Execute FFmpeg with libvmaf filter
        cmd = `$ffmpeg_path -hide_banner -v error -r 30 -i $dist_file -r 30 -i $ref_file -lavfi "libvmaf=model=version=vmaf_v0.6.1:log_path=$log_file:log_fmt=json" -f null -`
        run(cmd)

        # 3. Extract the mean score from the output log
        json_content = read(log_file, String)
        matches = collect(eachmatch(r"\"mean\":\s*([\d\.]+)", json_content))
        
        if !isempty(matches)
            return parse(Float64, matches[end].captures[1])
        else
            return NaN
        end

    catch e
        println("VMAF Calculation Failed: $e")
        return NaN
    finally
        rm(ref_file, force=true)
        rm(dist_file, force=true)
        rm(log_file, force=true)
    end
end