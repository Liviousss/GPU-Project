#pragma once
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <vector>
#include <iostream>

#define N_STREAMS 10

/**
 * GPU kernel wrapper. 
 * @param video base video in bytes.
 * @param blurred_video the blurred video in bytes returned by the kernel.
 * @param gaussianMatrix the gaussian matrix in 1D format.
 * @param DIM the base video lenght.
 * @param kernel_size the gaussian function kernel size.
 * @param height the video height.
 * @param width the video width.
 * @param channels the video channels.
 * @param dataTransferTime Pointer to the data transfer time time CPU-GPU and viceversa.
 * @param computationTime Pointer to the GPU computation time. 
*/
void kernel(unsigned char *frame, 
            unsigned char* blurred_frame, 
            float *gaussianMatrix, 
            unsigned int DIM, 
            int kernel_size, 
            int rows, 
            int columns, 
            int channels,
            int *dataTransferTime,
            int *computationTime);
