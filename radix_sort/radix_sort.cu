/**
 * CS5658: GPU PROGRAMMING — Group 18
 * Parallel 4-bit LSD Radix Sort in NVIDIA CUDA
 */

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// ============================================================================
// Constants
// ============================================================================
#define BLOCK_SIZE        256
#define RADIX_BITS        4
#define RADIX_BUCKETS     (1 << RADIX_BITS)   // 16
#define SCAN_BLOCK_SIZE   256
#define SCAN_ELEMS_PER_BLOCK (2 * SCAN_BLOCK_SIZE)
#define LOG_NUM_BANKS     5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << " — " << cudaGetErrorString(err) << "\n";        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// ============================================================================
// Kernel 1: Shared-Memory Pre-Sort + Local Histogram
// ============================================================================
__global__ void local_sort_and_histogram_kernel(
    const uint32_t* __restrict__ d_in,
    uint32_t* __restrict__ d_local_sorted,
    uint32_t* __restrict__ d_block_hist,
    int n, int bit)
{
    __shared__ uint32_t s_keys[BLOCK_SIZE];
    __shared__ int s_warp_zeros[BLOCK_SIZE / 32];
    __shared__ int s_warp_offset[BLOCK_SIZE / 32];
    __shared__ int s_total_zeros;
    __shared__ int s_digit_start[RADIX_BUCKETS];
    __shared__ int s_digit_end[RADIX_BUCKETS];

    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * BLOCK_SIZE;
    const int gid = block_start + tid;
    const int valid_count = (block_start + BLOCK_SIZE <= n) ? BLOCK_SIZE : (n - block_start);
    bool valid = (tid < valid_count);

    uint32_t key = valid ? d_in[gid] : 0xFFFFFFFFu;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    // 4-bit digit sort via 4 sequential 1-bit stable splits
    #pragma unroll
    for (int sub = 0; sub < RADIX_BITS; ++sub) {
        int pred = valid ? (((key >> (bit + sub)) & 1u) == 0u) : 0;

        unsigned int mask = __ballot_sync(0xFFFFFFFFu, pred);
        int warp_zeros  = __popc(mask);
        int prefix_zeros = __popc(mask & ((1u << lane) - 1u));

        if (lane == 0) s_warp_zeros[warp_id] = warp_zeros;
        __syncthreads();

        if (tid < (BLOCK_SIZE / 32)) {
            int val = s_warp_zeros[tid];
            int inclusive = val;
            #pragma unroll
            for (int delta = 1; delta < (BLOCK_SIZE / 32); delta <<= 1) {
                int other = __shfl_up_sync(0xFFu, inclusive, delta);
                if (tid >= delta) inclusive += other;
            }
            s_warp_offset[tid] = inclusive - val;
            if (tid == (BLOCK_SIZE / 32) - 1) s_total_zeros = inclusive;
        }
        __syncthreads();

        int zero_rank = s_warp_offset[warp_id] + prefix_zeros;
        int dest = pred ? zero_rank : (s_total_zeros + (tid - zero_rank));
        s_keys[dest] = key;
        __syncthreads();

        key = s_keys[tid];
        valid = (tid < valid_count);
        __syncthreads();
    }

    if (valid) d_local_sorted[gid] = key;

    if (tid < RADIX_BUCKETS) { s_digit_start[tid] = 0; s_digit_end[tid] = 0; }
    __syncthreads();

    if (valid) {
        int my_digit   = static_cast<int>((key >> bit) & 0xFu);
        int prev_digit = (tid > 0) ? static_cast<int>((s_keys[tid - 1] >> bit) & 0xFu) : -1;
        int next_digit = (tid + 1 < valid_count) ? static_cast<int>((s_keys[tid + 1] >> bit) & 0xFu) : RADIX_BUCKETS;
        if (my_digit != prev_digit) s_digit_start[my_digit] = tid;
        if (my_digit != next_digit) s_digit_end[my_digit]   = tid + 1;
    }
    __syncthreads();

    if (tid < RADIX_BUCKETS) {
        d_block_hist[tid * gridDim.x + blockIdx.x] =
            static_cast<uint32_t>(s_digit_end[tid] - s_digit_start[tid]);
    }
}

// ============================================================================
// Kernel 2: Blelloch Exclusive Prefix Sum (bank-conflict-free)
// ============================================================================
__global__ void blelloch_scan_kernel(
    const uint32_t* __restrict__ d_in,
    uint32_t* __restrict__ d_out,
    uint32_t* __restrict__ d_block_sums,
    int length)
{
    __shared__ uint32_t s[SCAN_ELEMS_PER_BLOCK + CONFLICT_FREE_OFFSET(SCAN_ELEMS_PER_BLOCK)];
    const int tid = threadIdx.x;
    const int base = blockIdx.x * SCAN_ELEMS_PER_BLOCK;
    const int ai = tid, bi = tid + SCAN_BLOCK_SIZE;
    const int boff_a = CONFLICT_FREE_OFFSET(ai), boff_b = CONFLICT_FREE_OFFSET(bi);

    s[ai + boff_a] = (base + ai < length) ? d_in[base + ai] : 0u;
    s[bi + boff_b] = (base + bi < length) ? d_in[base + bi] : 0u;

    int offset = 1;
    for (int d = SCAN_ELEMS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int a = offset * (2 * tid + 1) - 1 + CONFLICT_FREE_OFFSET(offset * (2 * tid + 1) - 1);
            int b = offset * (2 * tid + 2) - 1 + CONFLICT_FREE_OFFSET(offset * (2 * tid + 2) - 1);
            s[b] += s[a];
        }
        offset <<= 1;
    }

    if (tid == 0) {
        int last = (SCAN_ELEMS_PER_BLOCK - 1) + CONFLICT_FREE_OFFSET(SCAN_ELEMS_PER_BLOCK - 1);
        if (d_block_sums) d_block_sums[blockIdx.x] = s[last];
        s[last] = 0u;
    }

    for (int d = 1; d < SCAN_ELEMS_PER_BLOCK; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int a = offset * (2 * tid + 1) - 1 + CONFLICT_FREE_OFFSET(offset * (2 * tid + 1) - 1);
            int b = offset * (2 * tid + 2) - 1 + CONFLICT_FREE_OFFSET(offset * (2 * tid + 2) - 1);
            uint32_t t = s[a]; s[a] = s[b]; s[b] += t;
        }
    }
    __syncthreads();

    if (base + ai < length) d_out[base + ai] = s[ai + boff_a];
    if (base + bi < length) d_out[base + bi] = s[bi + boff_b];
}

__global__ void add_block_sums_kernel(uint32_t* d_data,
                                      const uint32_t* d_block_sums,
                                      int length)
{
    const int base = blockIdx.x * SCAN_ELEMS_PER_BLOCK;
    const uint32_t add = d_block_sums[blockIdx.x];
    if (base + threadIdx.x < length)                      d_data[base + threadIdx.x] += add;
    if (base + threadIdx.x + SCAN_BLOCK_SIZE < length)    d_data[base + threadIdx.x + SCAN_BLOCK_SIZE] += add;
}

// ============================================================================
// Kernel 3: Coalesced Scatter
// ============================================================================
__global__ void scatter_kernel(
    const uint32_t* __restrict__ d_local_sorted,
    const uint32_t* __restrict__ d_block_hist,
    const uint32_t* __restrict__ d_global_offsets,
    uint32_t* __restrict__ d_out,
    int n, int bit)
{
    __shared__ uint32_t s_local_off[RADIX_BUCKETS];
    __shared__ uint32_t s_global_off[RADIX_BUCKETS];
    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * BLOCK_SIZE;
    const int valid_count = (block_start + BLOCK_SIZE <= n) ? BLOCK_SIZE : (n - block_start);

    if (tid < RADIX_BUCKETS) {
        uint32_t h = d_block_hist[tid * gridDim.x + blockIdx.x];
        s_global_off[tid] = d_global_offsets[tid * gridDim.x + blockIdx.x];
        uint32_t inc = h;
        #pragma unroll
        for (int delta = 1; delta < RADIX_BUCKETS; delta <<= 1) {
            uint32_t other = __shfl_up_sync(0xFFFFu, inc, delta);
            if (tid >= delta) inc += other;
        }
        s_local_off[tid] = inc - h;
    }
    __syncthreads();

    if (tid < valid_count) {
        uint32_t key    = d_local_sorted[block_start + tid];
        uint32_t digit  = (key >> bit) & 0xFu;
        uint32_t rank   = static_cast<uint32_t>(tid) - s_local_off[digit];
        d_out[s_global_off[digit] + rank] = key;
    }
}

// Sign-bit flip for signed integer support
__global__ void flip_sign_bit_kernel(uint32_t* d_data, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) d_data[gid] ^= 0x80000000u;
}

// ============================================================================
// Multi-level Scan
// ============================================================================
struct ScanWorkspace {
    std::vector<uint32_t*> level_sums;

    void allocate(int length) {
        free_all();
        int cur = length;
        while (cur > SCAN_ELEMS_PER_BLOCK) {
            int nb = (cur + SCAN_ELEMS_PER_BLOCK - 1) / SCAN_ELEMS_PER_BLOCK;
            uint32_t* p = nullptr;
            CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(nb) * sizeof(uint32_t)));
            level_sums.push_back(p);
            cur = nb;
        }
    }
    void free_all() { for (auto p : level_sums) if (p) cudaFree(p); level_sums.clear(); }
    ~ScanWorkspace() { free_all(); }
};

void exclusive_scan_gpu(const uint32_t* d_in, uint32_t* d_out,
                        int length, ScanWorkspace& ws, int level = 0)
{
    int nb = (length + SCAN_ELEMS_PER_BLOCK - 1) / SCAN_ELEMS_PER_BLOCK;
    if (nb <= 1) {
        blelloch_scan_kernel<<<1, SCAN_BLOCK_SIZE>>>(d_in, d_out, nullptr, length);
        CUDA_CHECK(cudaGetLastError()); return;
    }
    uint32_t* d_sums = ws.level_sums[level];
    blelloch_scan_kernel<<<nb, SCAN_BLOCK_SIZE>>>(d_in, d_out, d_sums, length);
    CUDA_CHECK(cudaGetLastError());
    exclusive_scan_gpu(d_sums, d_sums, nb, ws, level + 1);
    add_block_sums_kernel<<<nb, SCAN_BLOCK_SIZE>>>(d_out, d_sums, length);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// GPU Workspace + Sort Driver
// ============================================================================
struct RadixSortWorkspace {
    uint32_t *d_ping = nullptr, *d_pong = nullptr;
    uint32_t *d_local_sorted = nullptr;
    uint32_t *d_block_hist = nullptr, *d_global_offsets = nullptr;
    ScanWorkspace scan_ws;
    int allocated_n = 0;

    void ensure(int n) {
        if (n <= allocated_n) return;
        release();
        int nb = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int hl = RADIX_BUCKETS * nb;
        CUDA_CHECK(cudaMalloc(&d_ping,           static_cast<size_t>(n)  * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_pong,           static_cast<size_t>(n)  * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_local_sorted,   static_cast<size_t>(n)  * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_block_hist,     static_cast<size_t>(hl) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_global_offsets, static_cast<size_t>(hl) * sizeof(uint32_t)));
        scan_ws.allocate(hl);
        allocated_n = n;
    }
    void release() {
        for (auto p : {d_ping, d_pong, d_local_sorted, d_block_hist, d_global_offsets})
            if (p) cudaFree(p);
        d_ping = d_pong = d_local_sorted = d_block_hist = d_global_offsets = nullptr;
        scan_ws.free_all(); allocated_n = 0;
    }
    ~RadixSortWorkspace() { release(); }
};

void gpu_radix_sort_device(uint32_t* d_data, int n,
                           RadixSortWorkspace& ws, bool is_signed)
{
    if (n <= 1) return;
    int nb  = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int hl  = RADIX_BUCKETS * nb;

    if (is_signed) { flip_sign_bit_kernel<<<nb, BLOCK_SIZE>>>(d_data, n); CUDA_CHECK(cudaGetLastError()); }

    uint32_t *cur_in = d_data, *cur_out = ws.d_pong;
    for (int bit = 0; bit < 32; bit += RADIX_BITS) {
        local_sort_and_histogram_kernel<<<nb, BLOCK_SIZE>>>(cur_in, ws.d_local_sorted, ws.d_block_hist, n, bit);
        CUDA_CHECK(cudaGetLastError());
        exclusive_scan_gpu(ws.d_block_hist, ws.d_global_offsets, hl, ws.scan_ws);
        scatter_kernel<<<nb, BLOCK_SIZE>>>(ws.d_local_sorted, ws.d_block_hist, ws.d_global_offsets, cur_out, n, bit);
        CUDA_CHECK(cudaGetLastError());
        std::swap(cur_in, cur_out);
    }

    if (is_signed) { flip_sign_bit_kernel<<<nb, BLOCK_SIZE>>>(d_data, n); CUDA_CHECK(cudaGetLastError()); }
}

// ============================================================================
// Test Helpers
// ============================================================================
std::vector<int> randomArray(int n)
{
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(INT_MIN, INT_MAX);
    std::vector<int> arr(n);
    for (int& x : arr) x = dist(rng);
    return arr;
}

void print_gpu_info()
{
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::cout << "============================================================\n"
              << "GPU Device        : " << p.name << "\n"
              << "Compute Capability: " << p.major << "." << p.minor << "\n"
              << "SM Count          : " << p.multiProcessorCount << "\n"
              << "Global Memory     : " << std::fixed << std::setprecision(2)
              << static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0) << " GB\n"
              << "============================================================\n";
}

int runTest(const std::string& name, const std::vector<int>& input,
            RadixSortWorkspace& ws)
{
    const int n = static_cast<int>(input.size());
    const int PRINT_LIMIT = 32;

    std::vector<uint32_t> h_data(n);
    for (int i = 0; i < n; ++i) h_data[i] = static_cast<uint32_t>(input[i]);

    ws.ensure(n);

    // Total GPU time (H2D + kernels + D2H)
    cudaEvent_t ev_tot0, ev_tot1, ev_k0, ev_k1;
    CUDA_CHECK(cudaEventCreate(&ev_tot0)); CUDA_CHECK(cudaEventCreate(&ev_tot1));
    CUDA_CHECK(cudaEventCreate(&ev_k0));   CUDA_CHECK(cudaEventCreate(&ev_k1));

    CUDA_CHECK(cudaEventRecord(ev_tot0));
    CUDA_CHECK(cudaMemcpy(ws.d_ping, h_data.data(),
                          static_cast<size_t>(n) * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(ev_k0));
    gpu_radix_sort_device(ws.d_ping, n, ws, true);
    CUDA_CHECK(cudaEventRecord(ev_k1));
    CUDA_CHECK(cudaEventSynchronize(ev_k1));

    std::vector<uint32_t> h_out(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), ws.d_ping,
                          static_cast<size_t>(n) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_tot1));
    CUDA_CHECK(cudaEventSynchronize(ev_tot1));

    float kernel_ms = 0, total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_k0, ev_k1));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms,  ev_tot0, ev_tot1));
    for (auto e : {ev_tot0, ev_tot1, ev_k0, ev_k1}) cudaEventDestroy(e);

    // CPU reference
    std::vector<int> gpu_result(n);
    for (int i = 0; i < n; ++i) gpu_result[i] = static_cast<int>(h_out[i]);

    std::vector<int> cpu_ref = input;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::sort(cpu_ref.begin(), cpu_ref.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok = (gpu_result == cpu_ref);

    std::cout << "\n====================================\n"
              << "Test case: " << name << "\n"
              << "Array size: " << n << "\n"
              << "====================================\n";
    if (n <= PRINT_LIMIT) {
        std::cout << "Original array: ";
        for (int x : input)     std::cout << x << " ";
        std::cout << "\nCPU sorted array: ";
        for (int x : cpu_ref)   std::cout << x << " ";
        std::cout << "\nGPU sorted array: ";
        for (int x : gpu_result) std::cout << x << " ";
        std::cout << "\n";
    }
    std::cout << "CPU verification: PASS\n"
              << "GPU verification: " << (ok ? "PASS" : "FAIL") << "\n"
              << "Sorting verification: " << (ok ? "PASS" : "FAIL") << "\n"
              << std::fixed << std::setprecision(4)
              << "CPU std::sort time: " << cpu_ms    << " ms\n"
              << "GPU kernel time: "    << kernel_ms  << " ms\n"
              << "Total GPU time: "     << total_ms   << " ms\n"
              << "Kernel speedup: "     << cpu_ms / kernel_ms << "x\n"
              << "Total speedup: "      << cpu_ms / total_ms  << "x\n";

    return ok ? 1 : 0;
}

// ============================================================================
// Test Suite
// ============================================================================
int run_standard_test_suite()
{
    int passed = 0, total = 0;

    RadixSortWorkspace ws;
    ws.ensure(4194304);   // pre-allocate for largest test

    // Warm-up: eliminate cold driver/JIT overhead before timing
    { uint32_t h[2] = {1u, 0u};
      CUDA_CHECK(cudaMemcpy(ws.d_ping, h, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
      gpu_radix_sort_device(ws.d_ping, 2, ws, true);
      CUDA_CHECK(cudaDeviceSynchronize()); }

    auto R = [&](const std::string& n, const std::vector<int>& v) {
        passed += runTest(n, v, ws); total++;
    };

    std::cout << "\n===== CORRECTNESS TESTS =====\n";
    R("Unsorted array",     {8, 3, 7, 1, 9, 2, 6, 4});
    R("Already sorted",     {1, 2, 3, 4, 5, 6, 7, 8});
    R("Reverse sorted",     {8, 7, 6, 5, 4, 3, 2, 1});
    R("All equal",          {5, 5, 5, 5, 5, 5, 5, 5});
    R("Duplicates",         {4, 2, 4, 1, 2, 1, 4, 2});
    R("Negative numbers",   {-5, -8, -2, -1, -7, -3, -4, -6});
    R("Mixed numbers",      {-5, 8, 0, -2, 6, -1, 4, 3});
    R("Nearly sorted",      {1, 2, 3, 5, 4, 6, 7, 8});
    R("Alternating values", {1, 9, 1, 9, 1, 9, 1, 9});
    R("Extreme values",     {INT_MAX, INT_MIN, 0, -1, INT_MAX, INT_MIN, 1, 5});

    std::cout << "\n===== PERFORMANCE TESTS =====\n";
    for (int n : {16, 32, 128, 256, 512, 1024, 4096, 16384,
                  65536, 262144, 1048576, 4194304}) {
        R("Random array", randomArray(n));
    }

    std::cout << "\n====================================\n"
              << "          TEST SUMMARY\n"
              << "====================================\n"
              << "Total test cases: " << total  << "\n"
              << "Passed: "           << passed << "\n"
              << "Failed: "           << total - passed << "\n"
              << (passed == total ? "ALL TEST CASES PASSED\n" : "SOME TEST CASES FAILED\n");

    return (passed == total) ? EXIT_SUCCESS : EXIT_FAILURE;
}

// ============================================================================
// Main
// ============================================================================
int main()
{
    print_gpu_info();
    return run_standard_test_suite();
}
