#include "../header/cuda_kernel.cuh"


/**
 * GPU function for the Gaussian Blur filter
 * @param image base image in bytes.
 * @param blurred_image the blurred image in bytes returned by the kernel.
 * @param gaussianMatrix the gaussian matrix in 1D format.
 * @param kernel_size the gaussian function kernel size.
 * @param height the image height.
 * @param width the image width.
 * @param channels the image channels.
*/
__global__ void blurImage(unsigned char *image,unsigned char *blurred_image,float *gaussianMatrix,int kernel_size, int height, int width, int channels){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    float value = 0.0f;
    int half_kernel_size = kernel_size/2;
    int DIM = channels*height*width;

    if(idx>=DIM) return;


    float blurred_value = 0.0;

    for(int m=-half_kernel_size; m<=half_kernel_size;m++){
        for(int n=-half_kernel_size; n<=half_kernel_size;n++){
            int toAddIdx = (m*width + n)*channels;
            if ( ! (idx+toAddIdx >= DIM || idx+toAddIdx < 0)){              
                unsigned char value = image[idx+toAddIdx];
                float gaussianValue = gaussianMatrix[(m+half_kernel_size)*kernel_size + (n+half_kernel_size)];
                float valueXblur = value * gaussianValue;
                blurred_value += valueXblur;
            }
                
        }
    }
    
    int intValue = (int)(blurred_value);
    unsigned char unsignedCharValue = (unsigned char)intValue;
    blurred_image[idx] = unsignedCharValue;
    
}

void kernel(unsigned char *image, 
            unsigned char* blurred_image, 
            float *gaussianMatrix, 
            int DIM, 
            int kernel_size, 
            int height, 
            int width, 
            int channels,
            int *dataTransferTime,
            int *computationTime){

    // create the device vectors
    unsigned char *device_image, *device_blurred_image;
    float *device_gaussianFunction;

    int size = DIM * sizeof(unsigned char);
    int gaussianSize = kernel_size * sizeof(float);

    std::chrono::high_resolution_clock::time_point startTransferTime, stopTransferTime;
    startTransferTime = std::chrono::high_resolution_clock::now();

    //IMAGE malloc and host to device copy

    cudaMalloc((void **)&device_image,size);
    cudaMemcpy(device_image,image,size,cudaMemcpyHostToDevice);

    cudaMalloc((void **)&device_blurred_image,size);

    //GAUSSIAN FUNCTION malloc and host to device copy

    cudaMalloc((void **)&device_gaussianFunction,kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(device_gaussianFunction,gaussianMatrix,kernel_size * kernel_size * sizeof(float),cudaMemcpyHostToDevice);


    stopTransferTime = std::chrono::high_resolution_clock::now();

    std::chrono::milliseconds firstTransferTime = std::chrono::duration_cast<std::chrono::milliseconds>(stopTransferTime-startTransferTime);
     
    //GPU CODE

    int threadXblock = 1024;
    int blocks = (DIM + threadXblock - 1) / threadXblock;

    cudaEvent_t startComputationTime,stopComputationTime;
    cudaEventCreate(&startComputationTime);
    cudaEventCreate(&stopComputationTime);

    cudaEventRecord(startComputationTime,0);
    blurImage <<<blocks,threadXblock>>>(device_image,device_blurred_image,device_gaussianFunction,kernel_size,height,width,channels);
    cudaEventRecord(stopComputationTime,0);

    cudaDeviceSynchronize();

    startTransferTime = std::chrono::high_resolution_clock::now();


    // device to host copy
    cudaMemcpy(blurred_image,device_blurred_image,size,cudaMemcpyDeviceToHost);
    
    stopTransferTime = std::chrono::high_resolution_clock::now();

    std::chrono::milliseconds secondTransferTime = std::chrono::duration_cast<std::chrono::milliseconds>(stopTransferTime-startTransferTime);
    *dataTransferTime = static_cast<int>(firstTransferTime.count() + secondTransferTime.count());
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,startComputationTime,stopComputationTime);
    *computationTime = elapsedTime;

    cudaDeviceSynchronize();

    //free 
    cudaFree(device_blurred_image);
    cudaFree(device_image);
    cudaFree(device_gaussianFunction);
};