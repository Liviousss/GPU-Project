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

void kernel(unsigned char *image, unsigned char* blurred_image, float *gaussianFunction, int DIM, int kernel_size, int rows, int columns, int channels){
    unsigned char *device_image, *device_blurred_image;
    float *device_gaussianFunction;

    int size = DIM * sizeof(unsigned char);
    int gaussianSize = kernel_size * sizeof(float);

    //IMAGE

    cudaMalloc((void **)&device_image,size);
    cudaMemcpy(device_image,image,size,cudaMemcpyHostToDevice);

    cudaMalloc((void **)&device_blurred_image,size);

    //GAUSSIAN FUNCTION

    cudaMalloc((void **)&device_gaussianFunction,kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(device_gaussianFunction,gaussianFunction,kernel_size * kernel_size * sizeof(float),cudaMemcpyHostToDevice);
     
    //GPU CODE

    int threadXblock = 1024;
    int blocksPerGrid = (DIM + threadXblock - 1) / threadXblock;

    blurImage <<<blocksPerGrid,threadXblock>>>(device_image,device_blurred_image,device_gaussianFunction,kernel_size,rows,columns,channels);

    cudaDeviceSynchronize();

    cudaMemcpy(blurred_image,device_blurred_image,size,cudaMemcpyDeviceToHost);
    

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