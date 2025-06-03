#pragma once

#include <math.h>
#include <vector>
#include "video.h"
#include "cuda_kernel.cuh"



class GaussianBlur{
    private:

        const int DEFAULT_KERNEL_SIZE = 7;
        const int DEFAULT_STD_DEV = 10;

        int kernel_size;
        int std_dev;

        int half_kernel_size;

        float *gaussianMatrix;

        void generateGaussianMatrix();

    
    public:

        GaussianBlur(){
            this->kernel_size = DEFAULT_KERNEL_SIZE;
            this->std_dev = DEFAULT_STD_DEV;
            this->half_kernel_size = (int)(this->kernel_size / 2);
            generateGaussianMatrix();
        }
       
        /**
         * Blur the video using the GPU.
         * @param video A Video object.
         * @param dataTransferTime Pointer to the data transfer time time CPU-GPU and viceversa.
         * @param computationTime Pointer to the GPU computation time. 
        */
        Video blurVideoGPU(Video video, int* dataTransferTime,int* computationTime);

        /**
         * Blur the video using the GPU cudaStreams.
         * @param video A Video object.
         * @param totalTime Pointer to the data transfer time plus computation time CPU-GPU and viceversa.
        */
        Video blurVideoGPUusingStreams(Video video, int* totalTime);

        /**
         * Blur the video using the GPU with shared memory.
         * @param video A Video object.
         * @param dataTransferTime Pointer to the data transfer time time CPU-GPU and viceversa.
         * @param computationTime Pointer to the GPU computation time. 
        */
        Video blurVideoGPUusingSharedMemory(Video video, int* dataTransferTime,int* computationTime);

        /**
         * Blur the video using the GPU cudaStreams combined with shared memory.
         * @param video A Video object.
         * @param totalTime Pointer to the data transfer time plus computation time CPU-GPU and viceversa.
        */
        Video blurVideoGPUusingSharedMemoryAndStreams(Video video, int* totalTime);


};