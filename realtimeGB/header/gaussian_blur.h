#pragma once

#include <math.h>
#include <vector>
#include "cuda_kernel.cuh"



class GaussianBlur{
    private:

        const int DEFAULT_KERNEL_SIZE = 15;
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
     * GPU function to blur a frame
     * @param frame an array of unsigned char.
     * @param width the frame width.
     * @param height the frame height.
     * @param channels the frame channels.
     * @param dataTransferTime Pointer to the data transfer time time CPU-GPU and viceversa.
     * @param computationTime Pointer to the GPU computation time.
    */
    unsigned char *blurFrame(unsigned char *frame, int width, int height, int channels, int* dataTransferTime,int* computationTime);
        
};