#include "../header/gaussian_blur.h"
#include <time.h>

float gaussianFunction(int x, int y, int std_dev){

    float std_dev_square = pow(std_dev,2);
    float coeff = 1 / (2 * M_PI * std_dev_square);
    float exp = - (x*x + y*y) / (2 * std_dev_square);

    float result = coeff * pow(M_E,exp);
    return result;
}


void GaussianBlur::generateGaussianMatrix(){
    float * gaussianMatrix = (float *)malloc(kernel_size * kernel_size * sizeof(float *));
    float sum = 0.0f;

    for(int i=0; i<kernel_size; i++){
        for(int j=0; j<kernel_size; j++){
            float value = gaussianFunction(i,j,std_dev);
            gaussianMatrix[i*kernel_size + j] = value;
            sum += value;
        }
    }

    //normalization
    for(int i=0; i<kernel_size; i++){
        for(int j=0; j<kernel_size; j++){
            gaussianMatrix[i*kernel_size + j] /= sum;
        }
    }


    this->gaussianMatrix = gaussianMatrix;
};

unsigned char *GaussianBlur::blurFrame(unsigned char *frame, int width, int height, int channels){
    
    int dim = width * height * channels;
    unsigned char * blurred_frame = (unsigned char *)malloc(dim * sizeof(unsigned char));

    kernel(frame,blurred_frame,gaussianMatrix,dim,kernel_size,height,width,channels);

    return blurred_frame;
}