#pragma once
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void kernel(unsigned char *image, 
            unsigned char* blurred_image, 
            float *gaussianFunction, 
            int DIM, 
            int kernel_size, 
            int rows, 
            int columns, 
            int channels,
            int *dataTransferTime,
            int *computationTime);

void testGPU();