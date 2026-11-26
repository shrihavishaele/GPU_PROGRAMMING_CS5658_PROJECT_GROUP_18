#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <climits>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Configuration Constants
constexpr int NUM_BUCKETS = 256;          // Number of buckets (splitters = NUM_BUCKETS - 1)
constexpr int OVERSAMPLE_FACTOR = 16;     // Oversampling multiplier
constexpr int BLOCK_SIZE = 256;           // Threads per block
constexpr int ELEMS_PER_THREAD = 4;       // Elements processed per thread in distribution
constexpr int TILE_SIZE = BLOCK_SIZE * ELEMS_PER_THREAD; // 1024 elements per block
constexpr int MAX_SHARED_BUCKET_SIZE = 2048; // Shared memory threshold before fallback

// Structure for Splitters with index tie-breaking for duplicate keys
struct Splitter {
    int key;
    int idx;
};

__host__ __device__ inline bool lessThan(int k1, int idx1, int k2, int idx2) {
    return (k1 < k2) || (k1 == k2 && idx1 < idx2);
}

// Binary search in splitters using (key, index) tie-breaking
__host__ __device__ int findBucket(int key, int idx, const Splitter* s_splitters) {
    int low = 0;
    int high = NUM_BUCKETS - 2; // 255 splitters for 256 buckets
    int bucket = NUM_BUCKETS - 1;

    while (low <= high) {
        int mid = (low + high) >> 1;
        if (lessThan(key, idx, s_splitters[mid].key, s_splitters[mid].idx)) {
            bucket = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return bucket;
}

// Kernel 1: Classify elements via shared-memory binary search & build local bucket histograms
__global__ void histogramAndClassifyKernel(
    const int* __restrict__ d_in,
    int* __restrict__ d_bucket_ids,
    int* __restrict__ d_local_offsets,
    int* __restrict__ d_block_histograms,
    const Splitter* __restrict__ d_splitters,
    int n,
    int num_blocks)
{
    __shared__ Splitter s_splitters[NUM_BUCKETS - 1];
    __shared__ int s_hist[NUM_BUCKETS];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Cooperatively load splitters into fast shared memory and zero local histogram
    if (tid < NUM_BUCKETS - 1) {
        s_splitters[tid] = d_splitters[tid];
    }
    if (tid < NUM_BUCKETS) {
        s_hist[tid] = 0;
    }
    __syncthreads();

    int base_idx = bid * TILE_SIZE + tid;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int idx = base_idx + i * BLOCK_SIZE;
        if (idx < n) {
            int val = d_in[idx];
            int b = findBucket(val, idx, s_splitters);
            int local_off = atomicAdd(&s_hist[b], 1);
            d_bucket_ids[idx] = b;
            d_local_offsets[idx] = local_off;
        }
    }
    __syncthreads();

    // Write local histogram in bucket-major layout so prefix scan computes global offsets
    if (tid < NUM_BUCKETS) {
        d_block_histograms[tid * num_blocks + bid] = s_hist[tid];
    }
}

// Kernel 2: Scatter elements to their exact global bucket positions
__global__ void scatterToBucketsKernel(
    const int* __restrict__ d_in,
    const int* __restrict__ d_bucket_ids,
    const int* __restrict__ d_local_offsets,
    const int* __restrict__ d_scanned_histograms,
    int* __restrict__ d_out,
    int n,
    int num_blocks)
{
    __shared__ int s_bucket_base[NUM_BUCKETS];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Load scanned global starting indices for each bucket in this block
    if (tid < NUM_BUCKETS) {
        s_bucket_base[tid] = d_scanned_histograms[tid * num_blocks + bid];
    }
    __syncthreads();

    int base_idx = bid * TILE_SIZE + tid;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int idx = base_idx + i * BLOCK_SIZE;
        if (idx < n) {
            int val = d_in[idx];
            int b = d_bucket_ids[idx];
            int local_off = d_local_offsets[idx];
            int dst = s_bucket_base[b] + local_off;
            d_out[dst] = val;
        }
    }
}

// Kernel 3: Sort each bucket locally inside shared memory (Odd-Even Transposition Sort)
__global__ void sortSmallBucketsKernel(
    int* __restrict__ d_data,
    const int* __restrict__ d_bucket_starts,
    const int* __restrict__ d_bucket_sizes)
{
    __shared__ int s_bucket[MAX_SHARED_BUCKET_SIZE];

    int b = blockIdx.x;
    int start = d_bucket_starts[b];
    int size = d_bucket_sizes[b];

    // Skip empty buckets or oversized buckets (handled by global sort fallback)
    if (size <= 1 || size > MAX_SHARED_BUCKET_SIZE) return;

    int tid = threadIdx.x;

    // Coalesced load of bucket elements into shared memory
    for (int i = tid; i < size; i += blockDim.x) {
        s_bucket[i] = d_data[start + i];
    }
    __syncthreads();

    // Parallel Odd-Even Transposition Sort in shared memory
    for (int phase = 0; phase < size; ++phase) {
        int parity = phase & 1;
        for (int i = 2 * tid + parity; i < size - 1; i += 2 * blockDim.x) {
            if (s_bucket[i] > s_bucket[i + 1]) {
                int tmp = s_bucket[i];
                s_bucket[i] = s_bucket[i + 1];
                s_bucket[i + 1] = tmp;
            }
        }
        __syncthreads();
    }

    // Coalesced write back to global memory
    for (int i = tid; i < size; i += blockDim.x) {
        d_data[start + i] = s_bucket[i];
    }
}

struct SampleSortStats {
    float kernel_ms = 0.0f;
    int fallback_buckets = 0;
};

// Select (NUM_BUCKETS - 1) splitters via random oversampling with index tie-breaking
std::vector<Splitter> selectSplitters(const std::vector<int>& data) {
    int n = static_cast<int>(data.size());
    int sample_count = NUM_BUCKETS * OVERSAMPLE_FACTOR;
    std::vector<Splitter> samples(sample_count);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, n - 1);

    for (int i = 0; i < sample_count; ++i) {
        int rand_idx = dist(rng);
        samples[i] = {data[rand_idx], rand_idx};
    }

    std::sort(samples.begin(), samples.end(), [](const Splitter& a, const Splitter& b) {
        return lessThan(a.key, a.idx, b.key, b.idx);
    });

    std::vector<Splitter> splitters(NUM_BUCKETS - 1);
    for (int i = 1; i < NUM_BUCKETS; ++i) {
        splitters[i - 1] = samples[i * OVERSAMPLE_FACTOR];
    }
    return splitters;
}

// Host driver for Sample Sort
void gpuSampleSort(std::vector<int>& h_data, SampleSortStats* stats = nullptr) {
    if (h_data.empty()) {
        if (stats != nullptr) {
            stats->kernel_ms = 0.0f;
            stats->fallback_buckets = 0;
        }
        return;
    }

    int n = static_cast<int>(h_data.size());
    int num_blocks = (n + TILE_SIZE - 1) / TILE_SIZE;

    // 1. Random Oversampling on Host to pick (NUM_BUCKETS - 1) Splitters
    std::vector<Splitter> h_splitters = selectSplitters(h_data);

    // 2. Allocate GPU Memory
    int *d_in, *d_out, *d_bucket_ids, *d_local_offsets;
    int *d_block_histograms, *d_scanned_histograms;
    int *d_bucket_starts, *d_bucket_sizes;
    Splitter* d_splitters;

    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bucket_ids, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_local_offsets, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_histograms, NUM_BUCKETS * num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scanned_histograms, NUM_BUCKETS * num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bucket_starts, NUM_BUCKETS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bucket_sizes, NUM_BUCKETS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_splitters, (NUM_BUCKETS - 1) * sizeof(Splitter)));

    CUDA_CHECK(cudaMemcpy(d_in, h_data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_splitters, h_splitters.data(), (NUM_BUCKETS - 1) * sizeof(Splitter), cudaMemcpyHostToDevice));

    // Create CUDA events for timing kernel execution
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(start_event));

    // 3. Step 1: Binary search classification and block-level bucket histograms
    histogramAndClassifyKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_in, d_bucket_ids, d_local_offsets, d_block_histograms, d_splitters, n, num_blocks
    );
    CUDA_CHECK(cudaGetLastError());

    // 4. Step 2: Parallel Exclusive Prefix Sum (Scan) over bucket histograms
    thrust::device_ptr<int> dev_hist_ptr(d_block_histograms);
    thrust::device_ptr<int> dev_scan_ptr(d_scanned_histograms);
    thrust::exclusive_scan(dev_hist_ptr, dev_hist_ptr + (NUM_BUCKETS * num_blocks), dev_scan_ptr);

    // 5. Step 3: Scatter elements to global buckets
    scatterToBucketsKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_in, d_bucket_ids, d_local_offsets, d_scanned_histograms, d_out, n, num_blocks
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute exact global start and size of each bucket for final sorting
    std::vector<int> h_scanned(NUM_BUCKETS * num_blocks);
    CUDA_CHECK(cudaMemcpy(h_scanned.data(), d_scanned_histograms, NUM_BUCKETS * num_blocks * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> h_bucket_starts(NUM_BUCKETS);
    std::vector<int> h_bucket_sizes(NUM_BUCKETS);
    for (int b = 0; b < NUM_BUCKETS; ++b) {
        h_bucket_starts[b] = h_scanned[b * num_blocks];
    }
    for (int b = 0; b < NUM_BUCKETS; ++b) {
        int next_start = (b + 1 < NUM_BUCKETS) ? h_bucket_starts[b + 1] : n;
        h_bucket_sizes[b] = next_start - h_bucket_starts[b];
    }

    CUDA_CHECK(cudaMemcpy(d_bucket_starts, h_bucket_starts.data(), NUM_BUCKETS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bucket_sizes, h_bucket_sizes.data(), NUM_BUCKETS * sizeof(int), cudaMemcpyHostToDevice));

    // 6. Step 4: Sort buckets in shared memory
    sortSmallBucketsKernel<<<NUM_BUCKETS, BLOCK_SIZE>>>(
        d_out, d_bucket_starts, d_bucket_sizes
    );
    CUDA_CHECK(cudaGetLastError());

    // 7. Fallback to global GPU sort for oversized buckets
    int fallback_count = 0;
    thrust::device_ptr<int> dev_out_ptr(d_out);
    for (int b = 0; b < NUM_BUCKETS; ++b) {
        if (h_bucket_sizes[b] > MAX_SHARED_BUCKET_SIZE) {
            fallback_count++;
            thrust::sort(dev_out_ptr + h_bucket_starts[b],
                         dev_out_ptr + h_bucket_starts[b] + h_bucket_sizes[b]);
        }
    }

    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));

    if (stats != nullptr) {
        stats->kernel_ms = ms;
        stats->fallback_buckets = fallback_count;
    } else {
        std::cout << "GPU Sample Sort Kernel + Scan Time: " << ms << " ms\n";
        std::cout << "Buckets sorted in Shared Memory: " << (NUM_BUCKETS - fallback_count)
                  << " / " << NUM_BUCKETS << " (Oversized Fallback Buckets: " << fallback_count << ")\n";
    }

    // Copy final sorted array back to host
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_bucket_ids));
    CUDA_CHECK(cudaFree(d_local_offsets));
    CUDA_CHECK(cudaFree(d_block_histograms));
    CUDA_CHECK(cudaFree(d_scanned_histograms));
    CUDA_CHECK(cudaFree(d_bucket_starts));
    CUDA_CHECK(cudaFree(d_bucket_sizes));
    CUDA_CHECK(cudaFree(d_splitters));
}

// CPU Sample Sort baseline (same oversampling, binary-search bucketing, scan, scatter, and bucket sort)
void cpuSampleSort(std::vector<int>& data) {
    if (data.size() <= 1) {
        return;
    }

    int n = static_cast<int>(data.size());
    std::vector<Splitter> splitters = selectSplitters(data);

    std::vector<int> bucket_ids(n);
    std::vector<int> counts(NUM_BUCKETS, 0);
    for (int i = 0; i < n; ++i) {
        int b = findBucket(data[i], i, splitters.data());
        bucket_ids[i] = b;
        counts[b]++;
    }

    std::vector<int> starts(NUM_BUCKETS, 0);
    for (int b = 1; b < NUM_BUCKETS; ++b) {
        starts[b] = starts[b - 1] + counts[b - 1];
    }

    std::vector<int> offsets = starts;
    std::vector<int> out(n);
    for (int i = 0; i < n; ++i) {
        out[offsets[bucket_ids[i]]++] = data[i];
    }

    for (int b = 0; b < NUM_BUCKETS; ++b) {
        if (counts[b] > 1) {
            std::sort(out.begin() + starts[b], out.begin() + starts[b] + counts[b]);
        }
    }

    data.swap(out);
}

// Generate a deterministic random integer array of size n
std::vector<int> randomArray(int n) {
    std::vector<int> arr(n);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 10000000);
    for (int i = 0; i < n; ++i) {
        arr[i] = dist(rng);
    }
    return arr;
}

// Run a single test case on both CPU Sample Sort and GPU Sample Sort and verify against std::sort
int runTest(const std::string& name, const std::vector<int>& input) {
    std::vector<int> expected = input;
    std::sort(expected.begin(), expected.end());

    std::vector<int> cpu_data = input;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuSampleSort(cpu_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::vector<int> gpu_data = input;
    SampleSortStats stats;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpuSampleSort(gpu_data, &stats);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_total_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    bool cpu_ok = (cpu_data == expected) && std::is_sorted(cpu_data.begin(), cpu_data.end());
    bool gpu_ok = (gpu_data == expected) && std::is_sorted(gpu_data.begin(), gpu_data.end());
    bool ok = cpu_ok && gpu_ok;

    double kernel_speedup = (stats.kernel_ms > 0.0f) ? (cpu_ms / stats.kernel_ms) : 0.0;
    double total_speedup = (gpu_total_ms > 0.0) ? (cpu_ms / gpu_total_ms) : 0.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << (ok ? "[PASS] " : "[FAIL] ")
              << name << " (n = " << input.size() << ")"
              << " | CPU: " << cpu_ms << " ms"
              << " | GPU Kernel+Scan: " << stats.kernel_ms << " ms"
              << " | GPU Total: " << gpu_total_ms << " ms"
              << std::setprecision(2)
              << " | Speedup(Kernel/Total): " << kernel_speedup << "x / " << total_speedup << "x"
              << " | Shared: " << (NUM_BUCKETS - stats.fallback_buckets)
              << "/" << NUM_BUCKETS
              << " (Fallback: " << stats.fallback_buckets << ")\n";

    if (!ok) {
        std::cout << "  CPU verification: " << (cpu_ok ? "PASS" : "FAIL")
                  << " | GPU verification: " << (gpu_ok ? "PASS" : "FAIL") << "\n";
        if (input.size() <= 16) {
            std::cout << "  Expected: ";
            for (int v : expected) std::cout << v << " ";
            std::cout << "\n  CPU Got:  ";
            for (int v : cpu_data) std::cout << v << " ";
            std::cout << "\n  GPU Got:  ";
            for (int v : gpu_data) std::cout << v << " ";
            std::cout << "\n";
        }
    }

    return ok ? 1 : 0;
}

int main() {
    // Initialize CUDA context ahead of timed tests
    CUDA_CHECK(cudaFree(0));

    int passed = 0;
    int total = 0;

    std::cout << "\nCORRECTNESS TESTS\n";

    // Test 1: Unsorted array
    passed += runTest("Unsorted array", {8, 3, 7, 1, 9, 2, 6, 4});
    total++;

    // Test 2: Already sorted array
    passed += runTest("Already sorted", {1, 2, 3, 4, 5, 6, 7, 8});
    total++;

    // Test 3: Reverse sorted array
    passed += runTest("Reverse sorted", {8, 7, 6, 5, 4, 3, 2, 1});
    total++;

    // Test 4: All equal elements
    passed += runTest("All equal", {5, 5, 5, 5, 5, 5, 5, 5});
    total++;

    // Test 5: Duplicate elements
    passed += runTest("Duplicates", {4, 2, 4, 1, 2, 1, 4, 2});
    total++;

    // Test 6: Negative numbers
    passed += runTest("Negative numbers", {-5, -8, -2, -1, -7, -3, -4, -6});
    total++;

    // Test 7: Mixed numbers
    passed += runTest("Mixed numbers", {-5, 8, 0, -2, 6, -1, 4, 3});
    total++;

    // Test 8: Nearly sorted array
    passed += runTest("Nearly sorted", {1, 2, 3, 5, 4, 6, 7, 8});
    total++;

    // Test 9: Alternating values
    passed += runTest("Alternating values", {1, 9, 1, 9, 1, 9, 1, 9});
    total++;

    // Test 10: Extreme integer values
    passed += runTest("Extreme values", {INT_MAX, INT_MIN, 0, -1,
                                         INT_MAX, INT_MIN, 1, 5});
    total++;

    std::cout << "\nPERFORMANCE TESTS\n";

    int sizes[] = {16, 32, 128, 256, 512, 1024, 4096, 16384,
                   65536, 262144, 1048576, 4194304};

    for (int n : sizes) {
        passed += runTest("Random array", randomArray(n));
        total++;
    }

    // Final test summary
    std::cout << "TEST SUMMARY\n";
    std::cout << "Total test cases: " << total << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << total - passed << std::endl;

    if (passed == total) {
        std::cout << "ALL TEST CASES PASSED" << std::endl;
        return EXIT_SUCCESS;
    } else {
        std::cout << "SOME TEST CASES FAILED" << std::endl;
        return EXIT_FAILURE;
    }
}
