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
    float ** gaussianMatrix = (float **)malloc(kernel_size * sizeof(float *));
    float sum = 0.0f;

    for(int i=0; i<kernel_size; i++){
        gaussianMatrix[i] = (float *)malloc(kernel_size * sizeof(float));
        for(int j=0; j<kernel_size; j++){
            float value = gaussianFunction(i,j,std_dev);
            gaussianMatrix[i][j] = value;
            sum += value;
        }
    }

    //normalizzazione
    for(int i=0; i<kernel_size; i++){
        for(int j=0; j<kernel_size; j++){
            gaussianMatrix[i][j] /= sum;
        }
    }


    this->gaussianMatrix = gaussianMatrix;
};

unsigned char *GaussianBlur::blurFrame(unsigned char *frame, int width, int height, int channels, int* dataTransferTime,int* computationTime){
    
    int dim = width * height * channels;
    unsigned char * blurred_frame = (unsigned char *)malloc(dim * sizeof(unsigned char));

    float *gaussianKernel = (float *)malloc(kernel_size*kernel_size*sizeof(float));

    for(int i=0;i<kernel_size*kernel_size;i++){
        gaussianKernel[i] = gaussianMatrix[i/kernel_size][i%kernel_size];
    }

    kernel(frame,blurred_frame,gaussianKernel,dim,kernel_size,height,width,channels,dataTransferTime,computationTime);

    return blurred_frame;
}