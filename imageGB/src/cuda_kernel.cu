#include "../header/cuda_kernel.cuh"

__global__ void blurImage(unsigned char *image,unsigned char *blurred_image,float *gaussianMatrix,int kernel_size, int imageRows, int imageColumns, int imageChannels){
    
    printf("qui arrivo 1");

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    float value = 0.0f;
    int channels = 3;
    int half_kernel_size = kernel_size/2;
    int DIM = imageChannels*imageRows*imageColumns;

    printf("qui arrivo 2");

    if(idx>=DIM) return;

    printf("qui arrivo");

    float blurred_value = 0.0;

    for(int m=-kernel_size; m<=kernel_size;m++){
        for(int n=-kernel_size; n<=kernel_size;n++){
            int toAddIdx = m*imageColumns*imageChannels + n*channels;
            if ( ! (idx+toAddIdx >= DIM || idx+toAddIdx < 0)){

                unsigned char value = image[idx+toAddIdx];
                float gaussianValue = gaussianMatrix[(m+half_kernel_size)*kernel_size + (n+half_kernel_size)];
                float valueXblur = value * gaussianValue;
                blurred_value += valueXblur;
            }
                
        }
    }
    
    int intValue = static_cast<int>(blurred_value);
    unsigned char unsignedCharValue = static_cast<unsigned char>(intValue);
    blurred_image[idx] = unsignedCharValue;
    
}

void kernel(unsigned char *image, unsigned char* blurred_image, float *gaussianFunction, int DIM, int kernel_size, int rows, int columns, int channels){
    unsigned char *device_image, *device_blurred_image;
    float *device_gaussianFunction;

    int size = DIM * sizeof(unsigned char);
    int gaussianSize = kernel_size * sizeof(float);

    //IMAGE

    cudaMalloc((void **)&device_image,size);
    cudaMemcpy(device_image,image,size,cudaMemcpyHostToDevice);

    //GAUSSIAN FUNCTION

    cudaMalloc((void **)&device_gaussianFunction,kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(device_gaussianFunction,gaussianFunction,kernel_size * kernel_size * sizeof(float),cudaMemcpyHostToDevice);
     
    //GPU CODE

    int threadXblock = 1024;
    int blocksPerGrid = (DIM + threadXblock - 1) / threadXblock;

    blurImage <<<1,1>>>(device_image,device_blurred_image,device_gaussianFunction,kernel_size,rows,columns,channels);

    cudaDeviceSynchronize();

    cudaMemcpy(blurred_image,device_blurred_image,size,cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

}