#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <iomanip>
#include <climits>
#include <string>
#include <cuda_runtime.h>
using namespace std;

void checkCuda(cudaError_t err){
    if (err != cudaSuccess){
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void bitonicSortKernel(int *data, int j, int k, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n){
        return;
    }

    // Only the first half of each comparison group is active
    if ((i / j) % 2 == 0){
        int partner = i + j;

        if (partner >= n){
            return;
        }

        // Alternate ascending and descending groups
        bool ascending;
        if ((i / k) % 2 == 0){
            ascending = true;
        }
        else{
            ascending = false;
        }

        // Ascending comparison
        if (ascending){
            if (data[i] > data[partner]){
                int temp = data[i];
                data[i] = data[partner];
                data[partner] = temp;
            }
        }
        // Descending comparison
        else{
            if (data[i] < data[partner]){
                int temp = data[i];
                data[i] = data[partner];
                data[partner] = temp;
            }
        }
    }
}

//launch kernels for each comparison stage
void bitonicSortGPU(int *d_data, int n){
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    for (int k = 2; k <= n; k = k * 2){
        for (int j = k / 2; j > 0; j = j / 2){
            bitonicSortKernel<<<blocks, threads>>>(d_data, j, k, n);
            checkCuda(cudaGetLastError());
        }
        if (k == n) break;
    }
}

//allocate memory, sort on GPU and measure execution time
void gpuSort(vector<int>& data, float& kernelTime, double& totalTime){
    int n = data.size();
    if (n <= 0 || (n & (n - 1)) != 0){
        cerr << "Invalid array size" << endl;
        exit(EXIT_FAILURE);
    }

    int *d_data = nullptr;
    size_t bytes = n * sizeof(int);
    auto totalStart = chrono::steady_clock::now();

    checkCuda(cudaMalloc((void**)&d_data, bytes));
    checkCuda(cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    checkCuda(cudaEventRecord(start));
    bitonicSortGPU(d_data, n);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&kernelTime, start, stop));

    checkCuda(cudaMemcpy(data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
    auto totalEnd = chrono::steady_clock::now();
    totalTime = chrono::duration<double, milli>(totalEnd - totalStart).count();

    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    checkCuda(cudaFree(d_data));
}

// Sequential CPU Bitonic Sort using the same comparison logic as the GPU
void bitonicSortCPU(vector<int>& data){
    int n = data.size();

    for (int k = 2; k <= n; k = k * 2){
        for (int j = k / 2; j > 0; j = j / 2){
            for (int i = 0; i < n; i++){
                if ((i / j) % 2 == 0){
                    int partner = i + j;
                    if (partner >= n){
                        continue;
                    }

                    bool ascending;
                    if ((i / k) % 2 == 0){
                        ascending = true;
                    }
                    else{
                        ascending = false;
                    }

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
        }
        if (k == n) break;
    }
}

void printArray(const vector<int>& arr){
    for (int x : arr){
        cout << x << " ";
    }
    cout << endl;
}

// Generate random array
vector<int> randomArray(int n){
    vector<int> arr(n);
    mt19937 rng(42);
    uniform_int_distribution<int> dist(0, 10000);

    for (int i = 0; i < n; i++){
        arr[i] = dist(rng);
    }
    return arr;
}

bool runTest(string name, vector<int> arr){
    int n = arr.size();
    cout << "\n====================================\n";
    cout << "Test case: " << name << endl;
    cout << "Array size: " << n << endl;
    cout << "====================================\n";

    if (n <= 32){
        cout << "Original array: ";
        printArray(arr);
    }

    vector<int> reference = arr;
    sort(reference.begin(), reference.end());

    vector<int> cpuData = arr;
    auto cpuStart = chrono::steady_clock::now();
    bitonicSortCPU(cpuData);
    auto cpuEnd = chrono::steady_clock::now();
    double cpuTime = chrono::duration<double, milli>(cpuEnd - cpuStart).count();

    float kernelTime = 0;
    double totalTime = 0;
    gpuSort(arr, kernelTime, totalTime);

    if (n <= 32){
        cout << "CPU sorted array: ";
        printArray(cpuData);
        cout << "GPU sorted array: ";
        printArray(arr);
    }

    bool cpuCorrect = (cpuData == reference);
    bool gpuCorrect = (arr == reference);
    bool correct = cpuCorrect && gpuCorrect;
    bool sorted = is_sorted(arr.begin(), arr.end());

    cout << "CPU verification: " << (cpuCorrect ? "PASS" : "FAIL") << endl;
    cout << "GPU verification: " << (gpuCorrect ? "PASS" : "FAIL") << endl;

    if (correct && sorted){
        cout << "Sorting verification: PASS" << endl;
    }
    else{
        cout << "Sorting verification: FAIL" << endl;
    }

    cout << fixed << setprecision(4);
    cout << "CPU Bitonic Sort time: " << cpuTime << " ms" << endl;
    cout << "GPU kernel time: " << kernelTime << " ms" << endl;
    cout << "Total GPU time: " << totalTime << " ms" << endl;

    if (kernelTime > 0){
        cout << "Kernel speedup: " << cpuTime / kernelTime << "x" << endl;
    }
    if (totalTime > 0){
        cout << "Total speedup: " << cpuTime / totalTime << "x" << endl;
    }
    return correct && sorted;
}

int main(int argc, char *argv[]){
    // Initialize CUDA before measuring execution time
    checkCuda(cudaFree(0));

    // Single input size for NSYS profiling
    if (argc == 2){
        int n = atoi(argv[1]);
        if (n <= 0 || (n & (n - 1)) != 0){
            cerr << "Invalid array size" << endl;
            return EXIT_FAILURE;
        }
        bool result = runTest("Random array", randomArray(n));
        if (result){
            return EXIT_SUCCESS;
        }
        else{
            return EXIT_FAILURE;
        }
    }

    int passed = 0;
    int total = 0;

    cout << "\nCORRECTNESS TESTS\n";

    //unsorted array
    passed += runTest("Unsorted array", {8, 3, 7, 1, 9, 2, 6, 4});
    total++;

    //already sorted array
    passed += runTest("Already sorted", {1, 2, 3, 4, 5, 6, 7, 8});
    total++;

    //reverse sorted array
    passed += runTest("Reverse sorted", {8, 7, 6, 5, 4, 3, 2, 1});
    total++;

    //all equal elements
    passed += runTest("All equal", {5, 5, 5, 5, 5, 5, 5, 5});
    total++;

    //duplicate elements
    passed += runTest("Duplicates", {4, 2, 4, 1, 2, 1, 4, 2});
    total++;

    //negative numbers
    passed += runTest("Negative numbers", {-5, -8, -2, -1, -7, -3, -4, -6});
    total++;

    //mixed numbers
    passed += runTest("Mixed numbers", {-5, 8, 0, -2, 6, -1, 4, 3});
    total++;

    //nearly sorted array
    passed += runTest("Nearly sorted", {1, 2, 3, 5, 4, 6, 7, 8});
    total++;

    //alternating values
    passed += runTest("Alternating values", {1, 9, 1, 9, 1, 9, 1, 9});
    total++;

    //extreme integer values
    passed += runTest("Extreme values", {INT_MAX, INT_MIN, 0, -1, INT_MAX, INT_MIN, 1, 5});
    total++;

    cout << "\n===== PERFORMANCE TESTS =====\n";

    int sizes[] = {16, 32, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304};

    for (int n : sizes){
        passed += runTest("Random array", randomArray(n));
        total++;
    }

    cout << "\n\n";
    cout << "TEST SUMMARY\n";

    cout << "Total test cases: " << total << endl;
    cout << "Passed: " << passed << endl;
    cout << "Failed: " << total - passed << endl;

    if (passed == total){
        cout << "ALL TEST CASES PASSED" << endl;
        return EXIT_SUCCESS;
    }
    else{
        cout << "SOME TEST CASES FAILED" << endl;
        return EXIT_FAILURE;
    }
}
