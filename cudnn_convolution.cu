#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits>

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }
#define CHECK_CUDNN(call) { cudnnStatus_t err = call; if (err != CUDNN_STATUS_SUCCESS) { printf("cuDNN error: %s\n", cudnnGetErrorString(err)); exit(1); } }


int main() {
    // Smaller, predefined sizes for human-readable output
    const int width = 1536;
    const int height = 2592;
    const int kernelSize = 5;
    const int inChannels = 1;
    const int outChannels = 1;
    const int batchSize = 1;
    const int inputSize = width * height * inChannels * batchSize;
    const int outputSize = width * height * outChannels * batchSize;
    const int kernelElements = kernelSize * kernelSize * inChannels * outChannels;
    const int filter_width = 5;
    const int filter_height = 5;

    std::cout << "Image size: " << width << "x" << height << "x" << inChannels << std::endl;
    std::cout << "Kernel size: " << kernelSize << "x" << kernelSize << "x" << inChannels << "x" << outChannels << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;

    // Allocate host memory
    float* h_input = (float*)malloc(inputSize * sizeof(float));
    float* h_output_cudnn = (float*)malloc(outputSize * sizeof(float));
    float* h_output_naive = (float*)malloc(outputSize * sizeof(float));

    srand(time(0));
    for (int i = 0; i < inputSize; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float h_kernel[filter_width * filter_height] = {
        -1.0f, 0.0f,  0.0f,  0.0f,  0.0f,
         0.0f, 0.0f,  0.0f,  0.0f,  0.0f,
         0.0f, 0.0f,  1.0f,  0.0f,  0.0f,
         0.0f, 0.0f,  0.0f,  0.0f,  0.0f,
         0.0f, 0.0f,  0.0f,  0.0f,  0.0f,
    };

    float *d_input, *d_kernel, *d_output_cudnn, *d_output_naive;
    CHECK_CUDA(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernelElements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_cudnn, outputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_naive, outputSize * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernelElements * sizeof(float), cudaMemcpyHostToDevice));

    // cuDNN setup
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernelDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, inChannels, height, width));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, outChannels, height, width));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outChannels, inChannels, kernelSize, kernelSize));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, kernelSize/2, kernelSize/2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Find the fastest cuDNN algorithm
    int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, inputDesc, kernelDesc, convDesc, outputDesc,
                                                       requestedAlgoCount, &returnedAlgoCount, perfResults));

    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
    for (int i = 1; i < returnedAlgoCount; i++) {
        std::cout << "Algorithm: " << perfResults[i].algo << " Time: " << perfResults[i].time << std::endl;
        if (perfResults[i].status == CUDNN_STATUS_SUCCESS && perfResults[i].time < perfResults[0].time) {
            algo = perfResults[i].algo;
        }
    }
    std::cout << "Selected algorithm: " << algo << std::endl;   
    size_t workspaceSize;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, algo, &workspaceSize));

    void* d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    // Warmup and benchmark runs
    const int warmupRuns = 5;
    const int benchmarkRuns = 20;
    float totalTime_cudnn = 0.0f;

    float alpha = 1.0f, beta = 0.0f;

    // Warmup runs
    for (int i = 0; i < warmupRuns; i++) {
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc,
                                            algo, d_workspace, workspaceSize, &beta, outputDesc, d_output_cudnn));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Benchmark runs
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < benchmarkRuns; i++) {
        // cuDNN benchmark
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc,
                                            algo, d_workspace, workspaceSize, &beta, outputDesc, d_output_cudnn));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime_cudnn += milliseconds;
    }

    // Calculate average times
    float avgTime_cudnn = totalTime_cudnn / benchmarkRuns;
    printf("cuDNN average time: %f ms\n", avgTime_cudnn);

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_output_cudnn, d_output_cudnn, outputSize * sizeof(float), cudaMemcpyDeviceToHost));


}