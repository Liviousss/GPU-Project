#pragma once

#include <math.h>
#include <vector>
#include "image.h"



class GaussianBlur{
    private:

        const int DEFAULT_KERNEL_SIZE = 7;
        const int DEFAULT_STD_DEV = 3;

        int kernel_size;
        int std_dev;

        int half_kernel_size;

        float **gaussianMatrix;

        void generateGaussianMatrix();

    
    public:

        GaussianBlur(){
            this->kernel_size = DEFAULT_KERNEL_SIZE;
            this->std_dev = DEFAULT_STD_DEV;
            this->half_kernel_size = (int)(this->kernel_size / 2);
            generateGaussianMatrix();
        }

        Image blurImage(Image image);



};