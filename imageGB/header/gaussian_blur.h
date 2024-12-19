#pragma once

#include <math.h>
#include <vector>
#include "image.h"



class GaussianBlur{
    private:

        const int DEFAULT_KERNEL_SIZE = 3;
        const int DEFAULT_STD_DEV = 2;

        int kernel_size;
        int std_dev;

        int half_kernel_size;

        std::vector<std::vector<float>> gaussianMatrix = {}; 
    
    public:

        GaussianBlur(){
            this->kernel_size = DEFAULT_KERNEL_SIZE;
            this->std_dev = DEFAULT_STD_DEV;
            this->half_kernel_size = (int)(this->kernel_size / 2);
            generateGaussianMatrix();
            int a=10;
        }

        

        std::vector<std::vector<float>> getGaussianMatrix();

        Image blurImage(Image image);

        
    
    private:
        void generateGaussianMatrix();
        




};