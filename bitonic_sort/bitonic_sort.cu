%%writefile bitonic_sort.cu
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

// CUDA error checking function
void checkCuda(cudaError_t err, int line){
    if (err != cudaSuccess){
        cerr << "CUDA Error: " << cudaGetErrorString(err)
             << " at line " << line << endl;
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCuda((call), __LINE__)


// GPU Bitonic Sort kernel
__global__ void bitonicSortKernel(int *data, int j, int k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n){
        return;
    }

    // Find the partner index using bitwise XOR
    int partner = i ^ j;

    // Only let the thread with the smaller index do the swap
    if (partner > i && partner < n){
        // Check if this block should be sorted in ascending order
        bool ascending = ((i & k) == 0);

        if (ascending){
            if (data[i] > data[partner]){
                int temp = data[i];
                data[i] = data[partner];
                data[partner] = temp;
            }
        }
        else{
            if (data[i] < data[partner]){
                int temp = data[i];
                data[i] = data[partner];
                data[partner] = temp;
            }
        }
    }
}


// Host function to perform Bitonic Sort on GPU
void bitonicSortGPU(int *d_data, int n){
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // k is the size of the current bitonic sequence
    for (int k = 2; k <= n; k = k * 2){
        // j is the distance between compared elements
        for (int j = k / 2; j > 0; j = j / 2){
            bitonicSortKernel<<<blocks, threadsPerBlock>>>(d_data, j, k, n);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}


// Check whether n is a power of two
bool isPowerOfTwo(int n){
    if (n <= 0){
        return false;
    }

    if ((n & (n - 1)) == 0){
        return true;
    }
    else{
        return false;
    }
}


// Main function
int main(int argc, char *argv[]){
    int n = 16;

    // Read array size from command line
    if (argc >= 2){
        n = atoi(argv[1]);
    }

    if (!isPowerOfTwo(n)){
        cerr << "Error: Array size must be a power of 2." << endl;
        cerr << "Example: ./bitonic_sort 1024" << endl;
        return EXIT_FAILURE;
    }

    cout << "\n=== GPU BITONIC SORT ===\n";
    cout << "Array size: " << n << endl;

    // 1. Generate random input on CPU
    vector<int> h_data(n);
    mt19937 rng(42);
    uniform_int_distribution<int> dist(0, 10000);

    for (int i = 0; i < n; i++){
        h_data[i] = dist(rng);
    }

    // Keep original input for verification
    vector<int> reference = h_data;
    sort(reference.begin(), reference.end());

    // Display original input for small arrays
    if (n <= 32){
        cout << "\nOriginal array:\n";
        for (int i = 0; i < n; i++){
            cout << h_data[i] << " ";
        }
        cout << endl;
    }

    // 2. Allocate GPU memory
    int *d_data = nullptr;
    int bytes = n * sizeof(int);

    auto totalStart = chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMalloc((void **)&d_data, bytes));

    // 3. Copy input from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

    // 4. Measure GPU kernel execution time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    bitonicSortGPU(d_data, n);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, stop));

    // 5. Copy sorted result from GPU to CPU
    vector<int> h_sorted(n);
    CUDA_CHECK(cudaMemcpy(h_sorted.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    auto totalEnd = chrono::high_resolution_clock::now();
    double totalTime = chrono::duration<double, milli>(totalEnd - totalStart).count();

    // 6. Verify sorting result
    cout << "\n==== RESULTS =====\n";

    bool correct = (h_sorted == reference);
    if (correct){
        cout << "Sorting verification: PASS" << endl;
    }
    else{
        cout << "Sorting verification: FAIL" << endl;
    }

    bool sorted = is_sorted(h_sorted.begin(), h_sorted.end());
    if (sorted){
        cout << "Ascending order check: PASS" << endl;
    }
    else{
        cout << "Ascending order check: FAIL" << endl;
    }

    // 7. Display sorted output for small arrays
    if (n <= 32){
        cout << "\nSorted array:\n";
        for (int i = 0; i < n; i++){
            cout << h_sorted[i] << " ";
        }
        cout << endl;
    }

    
    cout << fixed << setprecision(4);
    cout << "\n==== PERFORMANCE =====\n";
    cout << "GPU kernel time: " << kernelTime << " ms" << endl;
    cout << "Total GPU time (allocation, transfers, kernel): " << totalTime << " ms" << endl;

    // 8. Free GPU resources
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));

    if (correct){
        return EXIT_SUCCESS;
    }
    else{
        return EXIT_FAILURE;
    }
}