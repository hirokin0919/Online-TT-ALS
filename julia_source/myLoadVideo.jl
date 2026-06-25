"""
Load a sequence of grayscale images from a specified directory into a 3D tensor.

This function reads image files following the naming convention "in*.jpg", 
sorts them temporally, and stacks them into a 3D tensor format.

Inputs:
- folder_path: The path to the directory containing the frame images.

Returns:
- video_tensor: A 3D tensor of size (Height, Width, Time) containing grayscale values in [0.0, 1.0].
- height: The height of the video frames.
- width: The width of the video frames.
- T: The total number of frames (Time).
"""
function load_gray_video(folder_path::String)
    # Filter files that match the expected CDnet prefix/suffix and sort them to maintain temporal order
    frame_files = filter(f -> startswith(f, "in") && endswith(f, ".jpg"), readdir(folder_path))
    sort!(frame_files)

    T = length(frame_files)
    first_frame = load(joinpath(folder_path, frame_files[1]))
    height, width = size(first_frame)

    # Allocate a 3D tensor: (Height, Width, Time)
    video_tensor = Array{Float64, 3}(undef, height, width, T)

    for (t, file) in enumerate(frame_files)
        frame = load(joinpath(folder_path, file))
        # Convert the image to Grayscale and store it as Float64
        video_tensor[:, :, t] = Gray.(frame)
    end

    return video_tensor, height, width, T
end

"""
Export a 3D grayscale tensor as an MP4 video file.

Inputs:
- video_tensor: The 3D tensor to be exported, shaped (H, W, T).
- alg_name: The name of the algorithm used (e.g., "TT-FOA", "Original") for the title.
- ttrank: The TT-rank used for the approximation, displayed in the title.
- output_folder: The directory where the video will be saved.
- output_name: The filename of the output video (e.g., "output.mp4").
- myfps: The frames per second for the output video (default: 30).
"""
function save_gray_video(video_tensor::Array{Float64, 3}, alg_name::String, ttrank::Vector{Int}, output_folder::String, output_name::String, myfps::Int=30)
    T = size(video_tensor, 3)

    # Ensure the output directory exists
    if !isdir(output_folder)
        mkpath(output_folder)
    end

    # Generate an animation frame by frame
    # Note: yflip=true is required in Plots.heatmap to match standard image coordinate systems (origin at top-left)
    if alg_name == "Original"
        anim = @animate for t in 1:T
            frame = video_tensor[:, :, t]
            heatmap(frame, colorbar=false, axis=false, title="$(alg_name), Frame $t", clim=(0,1), c=:grays, yflip=true)
        end
    else
        anim = @animate for t in 1:T
            frame = video_tensor[:, :, t]
            heatmap(frame, colorbar=false, axis=false, title="$(alg_name) (r=$(ttrank)), Frame $t", clim=(0,1), c=:grays, yflip=true)
        end
    end
    mp4(anim, joinpath(output_folder, output_name), fps=myfps)
    sleep(1) # Brief pause to ensure file I/O is completed
end

"""
Save a specific spatial slice (frame) of a 3D grayscale tensor as an image file.
Useful for extracting qualitative visual comparisons for the paper.
"""
function save_gray_frame(video_tensor::Array{Float64,3}, output_folder::String, output_name::String, frame_idx::Int)
    if !isdir(output_folder)
        mkpath(output_folder)
    end

    frame = video_tensor[:, :, frame_idx]
    p = heatmap(frame, colorbar=false, axis=false, clim=(0,1), c=:grays, yflip=true)

    savefig(p, joinpath(output_folder, output_name))
end

"""
Load a sequence of color images from a specified directory into a 4D tensor.

Inputs:
- folder_path: The path to the directory containing the frame images.

Returns:
- video_tensor: A 4D tensor of size (Height, Width, 3, Time). The 3rd dimension corresponds to RGB channels.
- height: The height of the video frames.
- width: The width of the video frames.
- T: The total number of frames.
"""
function load_color_video(folder_path::String)
    frame_files = filter(f -> startswith(f, "in") && endswith(f, ".jpg"), readdir(folder_path))
    sort!(frame_files)

    T = length(frame_files)
    first_frame = load(joinpath(folder_path, frame_files[1]))
    height, width = size(first_frame)
    
    # Allocate a 4D tensor: (Height, Width, Channels (RGB), Time)
    video_tensor = Array{Float64, 4}(undef, height, width, 3, T)

    for (t, file) in enumerate(frame_files)
        frame = load(joinpath(folder_path, file))
        
        # Extract individual RGB channels and convert them to Float64 in the range [0.0, 1.0]
        video_tensor[:, :, 1, t] = Float64.(red.(frame))
        video_tensor[:, :, 2, t] = Float64.(green.(frame))
        video_tensor[:, :, 3, t] = Float64.(blue.(frame))
    end

    return video_tensor, height, width, T
end

"""
Export a 4D color tensor as an MP4 video file.
"""
function save_color_video(video_tensor::Array{Float64, 4}, alg_name::String, ttrank::Vector{Int}, output_folder::String, output_name::String, myfps::Int=30)
    H, W, C, T = size(video_tensor)

    # Ensure the output directory exists
    if !isdir(output_folder)
        mkpath(output_folder)
    end

    if alg_name == "Original"
        anim = @animate for t in 1:T
            frame = video_tensor[:, :, :, t]
            # Clamp values to prevent display errors from numerical approximation overshoots/undershoots
            clamp!(frame, 0.0, 1.0)

            # Reconstruct the RGB image object from the 3 color channels
            img = RGB.(frame[:, :, 1], frame[:, :, 2], frame[:, :, 3])

            plot(img, axis=false, ticks=false, title="$(alg_name), Frame $t")
        end
    else
        anim = @animate for t in 1:T
            frame = video_tensor[:, :, :, t]
            clamp!(frame, 0.0, 1.0)

            img = RGB.(frame[:, :, 1], frame[:, :, 2], frame[:, :, 3])

            plot(img, axis=false, ticks=false, title="$(alg_name) (r=$(ttrank)), Frame $t")
        end
    end

    output_path = joinpath(output_folder, output_name)
    mp4(anim, output_path, fps=myfps)
end

"""
Save a specific spatial slice (frame) of a 4D color tensor as an image file.
"""
function save_color_frame(video_tensor::Array{Float64,4}, output_folder::String, output_name::String, frame_idx::Int)
    if !isdir(output_folder)
        mkpath(output_folder)
    end

    frame = video_tensor[:, :, :, frame_idx]
    # Restrict values to [0.0, 1.0] for valid image rendering
    clamp!(frame, 0.0, 1.0)

    # Merge channels into a single RGB object
    img = RGB.(frame[:, :, 1], frame[:, :, 2], frame[:, :, 3])

    p = plot(img, axis=false, ticks=false)

    savefig(p, joinpath(output_folder, output_name))
end