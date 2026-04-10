# MNIST in CUDA


> Complete implementation progression from PyTorch to optimized CUDA: A step-by-step journey through 8 versions

## Purpose

This project implements a simple 2-layer MLP (Multi-Layer Perceptron) for MNIST digit classification, progressively optimizing from high-level PyTorch to low-level CUDA implementations. Each version demonstrates different optimization techniques and trade-offs between performance and complexity.

**Architecture**: 784 → 1024 → 10 (input → hidden → output)  
**Dataset**: 10,000 MNIST training samples, batch size 32, 10 epochs  
**Activation**: ReLU, Loss: Cross-entropy, Optimizer: SGD (lr=0.01)

## Setup
> DISCLAIMER: ensure you have a GPU with compute capability 5.0 or greater (at least maxwell architecture). See compatibility guide: https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html

```bash
git clone https://github.com/Infatoshi/mnist-cuda
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download MNIST data (if not already present)
python downloader.py
```

### CUDA Setup
For CUDA versions (v4-v5), ensure you have NVIDIA CUDA toolkit installed:
```bash
# Check CUDA installation
nvcc --version

# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Compile v4 (naive CUDA kernels) - use -arch=native or specify your GPU arch
nvcc -arch=native -o v4 v4.cu

# Compile v5 (cuBLAS optimized)
nvcc -arch=native -o v5 v5.cu -lcublas

```

> **Note**: The `-arch=native` flag ensures kernels are compiled for your specific GPU. Without it, custom CUDA kernels (like those in v4.cu) may produce incorrect results on newer GPUs. If `-arch=native` is not supported by your nvcc version, specify your architecture explicitly (e.g., `-arch=sm_86` for Ampere GPUs).


## Version Progression

### v1.py - PyTorch Baseline
- **Framework**: PyTorch with CUDA tensors
- **Features**: 
  - High-level PyTorch operations (Linear, ReLU, CrossEntropyLoss)
  - GPU tensors with automatic memory management
  - Built-in optimizations (cuDNN, etc.)
- **Purpose**: Establishes baseline performance and correctness reference

### v2.py - NumPy Implementation  
- **Framework**: Pure NumPy (CPU-only)
- **Features**:
  - Manual forward/backward pass implementation
  - Custom gradient computation and weight updates
  - He initialization for weights
- **Purpose**: Demonstrates the underlying math without GPU acceleration

### v3.c - C/CPU Implementation
- **Framework**: Pure C with timing breakdown
- **Features**:
  - Manual memory management
  - Detailed timing instrumentation per operation
  - CPU-optimized matrix operations
- **Purpose**: Shows CPU performance baseline and prepares for GPU porting

### v4.cu - Naive CUDA Kernels
- **Framework**: CUDA C with custom kernels
- **Features**:
  - Custom matrix multiplication kernels
  - Element-wise operations (ReLU, bias, softmax) on GPU
  - Manual memory transfers between host and device
- **Purpose**: First GPU implementation with basic CUDA kernels

### v5.cu - cuBLAS Optimized
- **Framework**: CUDA with cuBLAS library
- **Features**:
  - cuBLAS optimized matrix operations (SGEMM, SAXPY)
  - Persistent memory buffers to reduce allocations
  - Minimal host-device synchronization points
  - Optimized memory access patterns
- **Purpose**: Production-quality implementation with maximum performance

## Usage

```bash
# Run each version
python v1.py          # PyTorch baseline
python v2.py          # NumPy CPU implementation
gcc -o v3 v3.c -lm && ./v3                              # C CPU implementation
nvcc -arch=native -o v4 v4.cu && ./v4                   # Naive CUDA kernels
nvcc -arch=native -o v5 v5.cu -lcublas && ./v5          # Optimized cuBLAS
```

## Performance Comparison

Measured on RTX 3090 (Ampere architecture):

| Version | Implementation | Time | Speedup vs v3 | Final Loss |
|---------|----------------|------|---------------|------------|
| v1.py   | PyTorch CUDA   | ~0.5s | ~180x | 0.142 |
| v2.py   | NumPy CPU      | ~12s | ~8x | 0.142 |
| v3.c    | C CPU          | 90.6s | 1x (baseline) | 0.142 |
| v4.cu   | Naive CUDA     | 0.9s | 100x | 0.142 |
| v5.cu   | cuBLAS         | 0.4s | 225x | 0.142 |


### Timing Breakdown Analysis

Each implementation provides detailed timing breakdowns:

**v1 (PyTorch)**: High-level operations with cuDNN optimization
- Forward pass: ~40% of time
- Backward pass: ~50% of time  
- Weight updates: ~5% of time
- Data loading: ~5% of time

**v5 (cuBLAS Optimized)**: Production-level performance
- GPU compute (forward + backward + updates): ~60% of time
- Memory transfers (H2D + D2H): ~30% of time
- Host computation (loss calculation): ~10% of time



## Performance Insights

Key observations from the implementation progression:
- **Memory Management**: Persistent buffers (v5) vs per-batch allocation (v4) significantly impacts performance
- **Library Optimization**: cuBLAS provides highly optimized GEMM operations that outperform naive kernels
- **CPU-GPU Transfer**: Minimizing host-device synchronization is crucial for GPU performance
- **Numerical Stability**: Proper softmax implementation with max subtraction prevents overflow
- **Hardware Utilization**: v5 achieves the best performance by maximizing GPU compute utilization

## Key CUDA Concepts Demonstrated

- [Row vs Column Major](https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication): Matrix layout considerations for GEMM
- [Tensor Cores](https://docs.nvidia.com/cuda/cublas/#tensor-core-usage): Hardware acceleration for mixed-precision operations
- **Memory Patterns**: Coalesced access and persistent allocation strategies
- **Kernel Launch**: Grid/block dimensions and occupancy considerations

