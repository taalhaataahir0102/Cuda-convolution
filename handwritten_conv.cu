#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <time.h>
#include <chrono>

#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda_runtime.h>
#include <chrono>
#include <math.h>


#define TILE_WIDTH 16
#define MASK_COLS 5
#define MASK_ROWS 5
#define W (TILE_WIDTH + MASK_COLS - 1)

using namespace std;
using namespace std:: chrono;

// Mask in constant memory
__constant__ float deviceMaskData[MASK_ROWS * MASK_COLS];

__global__ void Convolution(float *inputImageData, float *outputImageData, int width, int height) {
    __shared__ float N_ds[W][W]; // Shared memory block

    int maskRadius = MASK_ROWS / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int dest = ty * TILE_WIDTH + tx;
    int destY = dest / W;
    int destX = dest % W;
    int srcY = by * TILE_WIDTH + destY - maskRadius;
    int srcX = bx * TILE_WIDTH + destX - maskRadius;

    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
        N_ds[destY][destX] = inputImageData[srcY * width + srcX];
    } else {
        N_ds[destY][destX] = 0.0f;
    }

    dest = ty * TILE_WIDTH + tx + TILE_WIDTH * TILE_WIDTH;
    destY = dest / W;
    destX = dest % W;
    srcY = by * TILE_WIDTH + destY - maskRadius;
    srcX = bx * TILE_WIDTH + destX - maskRadius;

    if (destY < W) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            N_ds[destY][destX] = inputImageData[srcY * width + srcX];
        } else {
            N_ds[destY][destX] = 0.0f;
        }
    }

    __syncthreads();

    // Perform convolution
    float accum = 0.0f;
    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        for (int y = 0; y < MASK_ROWS; y++) {
            for (int x = 0; x < MASK_COLS; x++) {
                accum += N_ds[ty + y][tx + x] * deviceMaskData[y * MASK_COLS + x];
            }
        }

        int y = by * TILE_WIDTH + ty;
        int x = bx * TILE_WIDTH + tx;
        if (y < height && x < width) {
            outputImageData[y * width + x] = accum;
        }
    }
}

int main() {
    const int imageWidth = 2592;
    const int imageHeight = 1536;
    const int imageSize = imageWidth * imageHeight;

    float *hostInputImageData = new float[imageSize];
    float *hostOutputImageData = new float[imageSize];
    float hostMaskData[MASK_ROWS * MASK_COLS] = {
        -1, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 1, 0, 0,
         0, 0, 0, 0, 0,
         0, 0, 0, 0, 0,
    };

    // Initialize random input image
    srand(time(0));
    for (int i = 0; i < imageSize; i++) {
        hostInputImageData[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *deviceInputImageData, *deviceOutputImageData;

    cudaMalloc((void**)&deviceInputImageData, imageSize * sizeof(float));
    cudaMalloc((void**)&deviceOutputImageData, imageSize * sizeof(float));

    cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceMaskData, hostMaskData, MASK_ROWS * MASK_COLS * sizeof(float));

    // Grid and block dimensions
    dim3 dimGrid(ceil(static_cast<float>(imageWidth) / TILE_WIDTH), ceil(static_cast<float>(imageHeight) / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    // Warmup and benchmark runs
    const int warmupRuns = 5;
    const int benchmarkRuns = 20;
    float totalTime_cudnn = 0.0f;

    // Warmup runs
    for (int i = 0; i < warmupRuns; i++) {
        Convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);
        cudaDeviceSynchronize();
    }

    // Measure time using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    for (int i = 0; i < benchmarkRuns; i++) {
        cudaEventRecord(start);
        Convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        totalTime_cudnn += milliseconds;
    }
    
    cout << "Kernel average execution time: " << totalTime_cudnn/benchmarkRuns << " ms\n";

    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] hostInputImageData;
    delete[] hostOutputImageData;
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
