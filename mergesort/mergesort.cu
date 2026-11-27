// parallel merge sort in cuda

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

#define block_size 256

// count elements in sorted arr that go before val
// left half uses <, right half uses <= to break ties
__device__ int count_smaller(const int* arr, int len, int val, bool is_left)
{
    int low = 0, high = len;
    while (low < high) {
        int mid = (low + high) / 2;
        if (arr[mid] < val || (!is_left && arr[mid] == val))
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

// step 1: sort each block of 256 elements in shared memory
__global__ void sort_blocks(int* data, int n)
{
    __shared__ int temp1[block_size];
    __shared__ int temp2[block_size];

    int tid = threadIdx.x;
    int i = blockIdx.x * block_size + tid;

    // pad extra slots with max int
    temp1[tid] = (i < n) ? data[i] : INT_MAX;
    __syncthreads();

    int* src = temp1;
    int* dst = temp2;

    for (int width = 1; width < block_size; width *= 2) {
        int pair_start  = (tid / (2 * width)) * (2 * width);
        int right_start = pair_start + width;
        int val = src[tid];
        int pos;

        if (tid < right_start)
            pos = (tid - pair_start) + count_smaller(src + right_start, width, val, true);
        else
            pos = (tid - right_start) + count_smaller(src + pair_start, width, val, false);

        dst[pair_start + pos] = val;
        __syncthreads();

        int* t = src; src = dst; dst = t;
    }

    if (i < n) data[i] = src[tid];
}

// step 2: merge sorted runs of size width across blocks
__global__ void merge_runs(const int* input, int* output, int n, int width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int pair_start  = (i / (2 * width)) * (2 * width);
    int right_start = pair_start + width;
    if (right_start > n) right_start = n;
    int right_end = pair_start + 2 * width;
    if (right_end > n) right_end = n;
    int val = input[i];
    int pos;

    if (i < right_start)
        pos = (i - pair_start) + count_smaller(input + right_start, right_end - right_start, val, true);
    else
        pos = (i - right_start) + count_smaller(input + pair_start, right_start - pair_start, val, false);

    output[pair_start + pos] = val;
}

// sort h_in on gpu and return kernel time in ms
float gpu_mergesort(const int* h_in, int* h_out, int n)
{
    if (n <= 1) {
        if (n == 1) h_out[0] = h_in[0];
        return 0.0f;
    }

    int *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMemcpy(d_a, h_in, n * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (n + block_size - 1) / block_size;

    // time only the parallel sorting kernels
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sort_blocks<<<blocks, block_size>>>(d_a, n);
    for (int width = block_size; width < n; width *= 2) {
        merge_runs<<<blocks, block_size>>>(d_a, d_b, n, width);
        int* t = d_a; d_a = d_b; d_b = t;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaMemcpy(h_out, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return gpu_ms;
}

// make a random array of size n (neg=true includes negative numbers)
vector<int> random_array(int n, bool neg = false)
{
    vector<int> arr(n);
    for (int i = 0; i < n; i++) arr[i] = neg ? (rand() - RAND_MAX / 2) : rand();
    return arr;
}

// run one test, compare with cpu sort, and return 1 if pass
int run_test(const char* name, vector<int> arr)
{
    int n = (int)arr.size();
    vector<int> out(n);

    float gpu_ms = gpu_mergesort(arr.data(), out.data(), n);

    clock_t c0 = clock();
    sort(arr.begin(), arr.end());
    double cpu_ms = 1000.0 * (clock() - c0) / CLOCKS_PER_SEC;

    int ok = (arr == out) ? 1 : 0;

    printf("%s (n=%d): gpu = %.5f ms, cpu = %.5f ms -> %s\n",
           name, n, gpu_ms, cpu_ms, ok ? "PASS" : "FAIL");

    return ok;
}

int main()
{
    srand(42);
    int passed = 0, total = 0;

    printf("\n===== TESTS =====\n");
    passed += run_test("Small array", {38, 27, 43, 3, 9, 82, 10, 5}); total++;

    // more duplicates
    vector<int> dup(1048576);
    for (int i = 0; i < (int)dup.size(); i++) dup[i] = i % 5;
    passed += run_test("More duplicates", dup); total++;

    // negative numbers included
    passed += run_test("Negative + mixed", random_array(1048576, true)); total++;

    // large arrays (up to 16m)
    int sizes[] = {262144, 1048576, 4194304, 16777216};
    for (int n : sizes) {
        passed += run_test("Large random", random_array(n));
        total++;
    }

    printf("\nPassed: %d / %d\n", passed, total);
    return (passed == total) ? 0 : 1;
}
