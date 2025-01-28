#pragma once
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <vector>
#include <iostream>

void kernel(unsigned char *frame, 
            unsigned char* blurred_frame, 
            float *gaussianFunction, 
            unsigned int DIM, 
            int kernel_size, 
            int rows, 
            int columns, 
            int channels,
            int *dataTransferTime,
            int *computationTime);


void kernelUsingStreams(unsigned char *frame, 
            unsigned char* blurred_frame, 
            float *gaussianFunction, 
            unsigned int DIM, 
            int kernel_size, 
            int rows, 
            int columns, 
            int channels,
            int *dataTransferTime,
            int *computationTime);


void testGPU();