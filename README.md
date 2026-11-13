# GPU_PROGRAMMING_CS5658_PROJECT_GROUP_18
# Parallel GPU Sorting Algorithms in CUDA

A high-performance implementation, optimization, and comparative study of four parallel sorting algorithms (**Radix Sort**, **Merge Sort**, **Sample Sort**, and **Bitonic Sort**) accelerated using **C++ and NVIDIA CUDA**.

---

## Team Members (Group 18)

| Name | Roll Number |
| :--- | :--- |
| **Shri Havish Aele** | `142301003` |
| **K Saideep** | `112301018` |
| **U Sai Vignesh** | `112301034` |
| **T G Sasi Kiran** | `112301033` |

---

## Abstract

Sorting millions of elements on a CPU is inherently bottlenecked by limited core counts and sequential memory access patterns. Offloading sorting to a GPU using NVIDIA CUDA unlocks massive parallelism across thousands of threads, but requires fundamentally redesigning standard algorithms to account for global memory bandwidth, shared memory constraints, and thread synchronization. 

This project implements, optimizes, and benchmarks four distinct GPU sorting architectures across both non-comparative and comparison-based paradigms.

---

## Implemented Algorithms

### 1. Radix Sort
* **Approach:** Sorts integers digit-by-digit (or bit-by-bit) across multiple passes without element-to-element comparisons.
* **CUDA Optimizations:**
  * Thread blocks load data chunks into fast shared memory to compute local digit histograms.
  * Uses a work-efficient parallel prefix sum (scan) to determine exact global output offsets.
  * Pre-sorts elements locally in shared memory using warp-level operations to avoid random global writes and guarantee **coalesced scatter writes**.

### 2. Merge Sort
* **Approach:** An iterative, bottom-up divide-and-conquer algorithm designed to eliminate recursion overhead on the GPU.
* **CUDA Optimizations:**
  * **Base Stage:** Individual CUDA thread blocks independently sort small subarrays entirely within low-latency shared memory.
  * **Merge Stages:** Adjacent sorted runs are merged in parallel across multiple thread blocks with careful memory management and synchronization.

### 3. Sample Sort
* **Approach:** A generalized multi-way quicksort that partitions the input into multiple buckets in a single global distribution pass using randomly oversampled pivot values (*splitters*).
* **CUDA Optimizations:**
  * Splitters are cached in shared memory, where threads perform binary search to classify elements into buckets.
  * Local bucket counts are aggregated via an exclusive prefix sum (scan) to compute global bucket offsets for coalesced writes, followed by local shared-memory sorting.
  * **Skew & Duplicate Handling:** Uses element indices for tie-breaking on duplicate keys and automatically falls back to a global sort for oversized buckets to prevent shared-memory overflow.

### 4. Bitonic Sort
* **Approach:** Maps a deterministic $O(N \log^2 N)$ sorting network of compare-and-swap operations directly onto CUDA threads.
* **CUDA Optimizations:**
  * **Kernel Fusion via Shared Memory:** Early compare-and-swap stages operate on contiguous local segments that fit completely inside shared memory, combining multiple steps into a single kernel launch to drastically reduce global memory traffic and launch overhead.
  * **Arbitrary Array Sizes:** Employs virtual padding and index masking so the sorter works seamlessly on any input size $N$, removing the strict power-of-two limitation.

---

## Key Learning Outcomes

* **Algorithmic Design:** End-to-end design and tuning of non-comparative (**Radix**) vs. comparison-based (**Merge, Sample, Bitonic**) parallel sorting algorithms in CUDA C++.
* **Parallel Primitives:** Hands-on implementation of foundational GPU building blocks, including shared-memory histograms, parallel prefix sums (scans), and parallel binary search.
* **Hardware-Level Optimization:** Maximizing throughput through global memory coalescing, shared-memory bank conflict avoidance, warp-level intrinsics, and branch divergence reduction.

---

## Prerequisites & Environment

* **GPU:** NVIDIA CUDA-capable GPU
* **Toolkit:** NVIDIA CUDA Toolkit (`nvcc`)
* **Compiler:** C++14 / C++17 compatible host compiler (`g++` or `clang++`)

---

## How to Build & Run

Clone the repository:
```bash
git clone <https://github.com/shrihavishaele/GPU_PROGRAMMING_CS5658_PROJECT_GROUP_18/tree/main/bitonic_sort >
cd <GPU_PROGRAMMING_CS5658_PROJECT_GROUP_18 >