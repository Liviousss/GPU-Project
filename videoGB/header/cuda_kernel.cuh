#pragma once
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <vector>
#include <iostream>

/**
 * GPU kernel wrapper. 
 * @param video base video in bytes.
 * @param blurred_video the blurred video in bytes returned by the kernel.
 * @param gaussianMatrix the gaussian matrix in 1D format.
 * @param DIM the base video lenght.
 * @param kernel_size the gaussian matrix kernel size.
 * @param height the video height.
 * @param width the video width.
 * @param channels the video channels.
 * @param frames the video frames.
 * @param dataTransferTime Pointer to the data transfer time time CPU-GPU and viceversa.
 * @param computationTime Pointer to the GPU computation time. 
*/
void kernel(unsigned char *video, 
            unsigned char* blurred_video, 
            float *gaussianMatrix, 
            unsigned int DIM, 
            int kernel_size, 
            int rows, 
            int columns, 
            int channels,
            int frames,
            int *dataTransferTime,
            int *computationTime);

/**
 * GPU kernel wrapper for the cudaStream implementation. 
 * @param video base video in bytes.
 * @param blurred_video the blurred video in bytes returned by the kernel.
 * @param gaussianMatrix the gaussian matrix in 1D format.
 * @param DIM the base video lenght.
 * @param kernel_size the gaussian matrix kernel size.
 * @param height the video height.
 * @param width the video width.
 * @param channels the video channels.
 * @param frames the video frames.
 * @param totalTime Pointer to the data transfer time plus computation time CPU-GPU and viceversa.
*/
void kernelUsingStreams(unsigned char *video, 
            unsigned char* blurred_video, 
            float *gaussianmatrix, 
            unsigned int DIM, 
            int kernel_size, 
            int rows, 
            int columns, 
            int channels,
            int frames,
            int *totalTime);

/**
 * GPU kernel wrapper for the cudaStream implementation. 
 * @param video base video in bytes.
 * @param blurred_video the blurred video in bytes returned by the kernel.
 * @param gaussianMatrix the gaussian matrix in 1D format.
 * @param DIM the base video lenght.
 * @param kernel_size the gaussian matrix kernel size.
 * @param height the video height.
 * @param width the video width.
 * @param channels the video channels.
 * @param frames the video frames.
 * @param dataTransferTime Pointer to the data transfer time time CPU-GPU and viceversa.
 * @param computationTime Pointer to the GPU computation time. 
*/
void kernelUsingSharedMemory(unsigned char *video, 
            unsigned char* blurred_video, 
            float *gaussianMatrix, 
            unsigned int DIM, 
            int kernel_size, 
            int rows, 
            int columns, 
            int channels,
            int frames,
            int *dataTransferTime,
            int *computationTime,
            bool *isPossibleToUseSharedMemory);