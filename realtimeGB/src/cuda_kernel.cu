#include "../header/cuda_kernel.cuh"

/**
 * GPU function for the Gaussian Blur filter
 * @param frame base frame in bytes.
 * @param blurred_frame the blurred frame in bytes returned by the kernel.
 * @param gaussianMatrix the gaussian matrix in 1D format.
 * @param kernel_size the gaussian function kernel size.
 * @param height the frame height.
 * @param width the frame width.
 * @param channels the frame channels.
*/
__global__ void blurframe(unsigned char *frame,unsigned char *blurred_frame,float *gaussianMatrix,int kernel_size, int frameRows, int frameColumns, int frameChannels){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    float value = 0.0f;
    int half_kernel_size = kernel_size/2;
    int DIM = frameChannels*frameRows*frameColumns;

    if(idx>=DIM) return;


    float blurred_value = 0.0;

    for(int m=-half_kernel_size; m<=half_kernel_size;m++){
        for(int n=-half_kernel_size; n<=half_kernel_size;n++){
            int toAddIdx = (m*frameColumns + n)*frameChannels;
            if ( ! (idx+toAddIdx >= DIM || idx+toAddIdx < 0)){              
                unsigned char value = frame[idx+toAddIdx];
                float gaussianValue = gaussianMatrix[(m+half_kernel_size)*kernel_size + (n+half_kernel_size)];
                float valueXblur = value * gaussianValue;
                blurred_value += valueXblur;
            }
                
        }
    }
    
    int intValue = (int)(blurred_value);
    unsigned char unsignedCharValue = (unsigned char)intValue;
    blurred_frame[idx] = unsignedCharValue;
    
}

void kernel(unsigned char *frame, 
            unsigned char* blurred_frame, 
            float *gaussianFunction, 
            unsigned int DIM, 
            int kernel_size, 
            int rows, 
            int columns, 
            int channels,
            int *dataTransferTime,
            int *computationTime){

    unsigned char *device_frame, *device_blurred_frame;
    float *device_gaussianFunction;

    int size = DIM * sizeof(unsigned char);
    int gaussianSize = kernel_size * sizeof(float);

    time_t startTransferTime;
    time(&startTransferTime);

    //frame

    cudaMalloc((void **)&device_frame,size);
    cudaMemcpy(device_frame,frame,size,cudaMemcpyHostToDevice);

    cudaMalloc((void **)&device_blurred_frame,size);

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
    blurframe <<<blocksPerGrid,threadXblock>>>(device_frame,device_blurred_frame,device_gaussianFunction,kernel_size,rows,columns,channels);
    cudaEventRecord(stopComputationTime,0);

    cudaError_t error = cudaGetLastError();

    cudaDeviceSynchronize();

    time(&startTransferTime);

    cudaMemcpy(blurred_frame,device_blurred_frame,size,cudaMemcpyDeviceToHost);
    
    time(&stopTransferTime);

    int secondTime = difftime(stopTransferTime,startTransferTime);
    *dataTransferTime = firstTime + secondTime;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,startComputationTime,stopComputationTime);
    *computationTime = elapsedTime;

    //Free Memory
    cudaFree(device_blurred_frame);
    cudaFree(device_frame);
    cudaFree(device_gaussianFunction);

    cudaDeviceSynchronize();
};