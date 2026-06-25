# Online-TT-ALS
Official Julia implementation of Online TT-ALS for scalable and exact streaming tensor decomposition.

This repository contains the official Julia implementation for the paper:  
**Online TT-ALS for Streaming Tensor Decomposition with Incremental Orthogonalization**

The repository provides the source code for the proposed **Online TT-ALS** algorithm, baseline algorithms (TT-FOA, Batch TT-ALS), and Jupyter notebooks to reproduce the quantitative and qualitative evaluations presented in the paper.

## Repository Structure

The code is organized into Jupyter notebooks for running experiments and a source directory containing the core algorithm implementations.

```text
.
├── ALS_color.ipynb               # Runs decomposition experiments on color videos (4D tensors)
├── ALS_gray.ipynb                # Runs decomposition experiments on grayscale videos (3D tensors)
├── ALS_synthetic.ipynb           # Runs scalability and ablation studies on synthetic data
├── ALS_indicator_color.ipynb     # Computes metrics (PSNR, VMAF, etc.) and generates LaTeX tables for color videos
├── ALS_indicator_gray.ipynb      # Computes metrics and generates LaTeX tables for grayscale videos
├── ALS_indicator_synthetic.ipynb # Computes metrics and generates LaTeX tables for synthetic data
│
└── julia_source/                 # Core Julia modules
    ├── myTTD.jl                  # Implementation of Proposed Online TT-ALS and Baselines (TT-FOA, Batch TT-ALS)
    ├── myGenerateData.jl         # Helper functions to generate synthetic TT streaming tensors
    ├── myLoadVideo.jl            # Helper functions for loading and exporting video frames/tensors
    └── myIndicator.jl            # Functions for computing evaluation metrics (RE, M-RMSE, SSIM, LPIPS, VMAF)
```

## Notebook Execution Flow

To reproduce the results for a specific experiment (e.g., the grayscale video evaluation), please execute the notebooks in the following order:

1. **Decomposition:** Run `ALS_gray.ipynb`. This script executes the proposed and baseline algorithms, and saves the reconstructed tensors as a `.jld2` file.
2. **Evaluation:** Run `ALS_indicator_gray.ipynb`. This notebook loads the generated `.jld2` file, evaluates the mathematical and perceptual metrics, and outputs the formatted LaTeX tables.

*(The same workflow applies to the color video and synthetic data experiments.)*

## Requirements

This codebase is written in **Julia** (tested on v1.11.x).

### 1. Julia Dependencies
Please install the required external Julia packages before running the notebooks. You can install them all at once via the Julia REPL (press `]` to enter the Pkg prompt):

```julia
julia> ]
pkg> add TensorToolbox Plots Images ImageIO FileIO FFMPEG_jll Revise JLD2 ImageQualityIndexes PyCall
```
*(Note: Standard libraries like `LinearAlgebra`, `Random`, `Statistics`, and `Printf` are already included in Julia and do not need to be installed.)*

### 2. External Dependencies
To calculate the advanced perceptual metrics (VMAF and LPIPS) in the `indicator` notebooks, the following external tools must be installed on your system:
* **FFmpeg (with libvmaf):** Required for VMAF calculation. Please update the `ffmpeg_path` variable in `julia_source/myIndicator.jl` to match your local installation.
* **Python & PyTorch:** Required for LPIPS calculation. The `lpips` python package must be accessible from your Julia environment via `PyCall`.

## Datasets

Due to file size limitations and licensing constraints, the raw video datasets are not included in this repository. 
To reproduce the video experiments, please download the **"office"**, **"pedestrians"**, and **"continuousPan"** sequences from the [CDnet 2014 benchmark](http://www.changedetection.net/). 
Place the extracted frame images in their respective input directories as specified in the notebooks (e.g., `../CDNet/office/input/`, `../CDNet/pedestrians/input/`, and `../CDNet/continuousPan/input/`).
