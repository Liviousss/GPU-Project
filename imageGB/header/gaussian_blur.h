#pragma once

#include <math.h>
#include <vector>
#include "image.h"
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
         * Blur the image using the CPU.
         * @param image An Image object.
         * @param duration Pointer to the function computation time. 
        */
        Image blurImage(Image image, int* duration);

        /**
         * Blur the image using the GPU.
         * @param image An Image object.
         * @param dataTransferTime Pointer to the data transfer time time CPU-GPU and viceversa.
         * @param computationTime Pointer to the GPU computation time. 
        */
        Image blurImageGPU(Image image, int* dataTransferTime,int* computationTime);



};