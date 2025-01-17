#include "../header/cuda_kernel.cuh"

__global__ void helloFromGPU (void) {
    printf("Hello World from Jetson GPU!\n");
}


__global__ void blurImage(unsigned char *image,unsigned char *blurred_image,float *gaussianMatrix,int kernel_size, int imageRows, int imageColumns, int imageChannels){
    
    

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    float value = 0.0f;
    int half_kernel_size = kernel_size/2;
    int DIM = imageChannels*imageRows*imageColumns;

    if(idx>=DIM) return;


    float blurred_value = 0.0;

    for(int m=-half_kernel_size; m<=half_kernel_size;m++){
        for(int n=-half_kernel_size; n<=half_kernel_size;n++){
            int toAddIdx = (m*imageColumns + n)*imageChannels;
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
            float *gaussianFunction, 
            int DIM, 
            int kernel_size, 
            int rows, 
            int columns, 
            int channels,
            int *dataTransferTime,
            int *computationTime){

    unsigned char *device_image, *device_blurred_image;
    float *device_gaussianFunction;

    int size = DIM * sizeof(unsigned char);
    int gaussianSize = kernel_size * sizeof(float);

    time_t startTransferTime;
    time(&startTransferTime);

    //IMAGE

    cudaMalloc((void **)&device_image,size);
    cudaMemcpy(device_image,image,size,cudaMemcpyHostToDevice);

    cudaMalloc((void **)&device_blurred_image,size);

    //GAUSSIAN FUNCTION

    cudaMalloc((void **)&device_gaussianFunction,kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(device_gaussianFunction,gaussianFunction,kernel_size * kernel_size * sizeof(float),cudaMemcpyHostToDevice);


    time_t stopTransferTime;
    time(&stopTransferTime);

    int firstTime = difftime(stopTransferTime,startTransferTime);
     
    //GPU CODE

    int threadXblock = 1024;
    int blocksPerGrid = (DIM + threadXblock - 1) / threadXblock;

    cudaEvent_t startComputationTime,stopComputationTime;
    cudaEventCreate(&startComputationTime);
    cudaEventCreate(&stopComputationTime);

    cudaEventRecord(startComputationTime,0);
    blurImage <<<blocksPerGrid,threadXblock>>>(device_image,device_blurred_image,device_gaussianFunction,kernel_size,rows,columns,channels);
    cudaEventRecord(stopComputationTime,0);

    cudaDeviceSynchronize();

    time(&startTransferTime);

    cudaMemcpy(blurred_image,device_blurred_image,size,cudaMemcpyDeviceToHost);
    
    time(&stopTransferTime);

    int secondTime = difftime(stopTransferTime,startTransferTime);
    *dataTransferTime = firstTime + secondTime;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,startComputationTime,stopComputationTime);
    *computationTime = elapsedTime;

    cudaDeviceSynchronize();
};

void testGPU(){
    printf("Hello World from CPU!\n");
    helloFromGPU <<<1, 10>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceReset();

}