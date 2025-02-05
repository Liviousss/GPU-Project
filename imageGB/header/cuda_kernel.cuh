#pragma once
#include <stdio.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

/**
 * GPU kernel wrapper. 
 * @param image base image in bytes.
 * @param blurred_image the blurred image in bytes returned by the kernel.
 * @param gaussianMatrix the gaussian matrix in 1D format.
 * @param DIM the base image lenght.
 * @param kernel_size the gaussian function kernel size.
 * @param height the image height.
 * @param width the image width.
 * @param channels the image channels.
 * @param dataTransferTime Pointer to the data transfer time time CPU-GPU and viceversa.
 * @param computationTime Pointer to the GPU computation time. 
*/
void kernel(unsigned char *image, 
            unsigned char* blurred_image, 
            float *gaussianMatrix, 
            int DIM, 
            int kernel_size, 
            int height, 
            int width, 
            int channels,
            int *dataTransferTime,
            int *computationTime);