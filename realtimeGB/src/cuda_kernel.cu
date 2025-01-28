#include "../header/cuda_kernel.cuh"

__global__ void helloFromGPU (void) {
    printf("Hello World from Jetson GPU!\n");
}


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

    cudaDeviceSynchronize();
};

void kernelUsingStreams(unsigned char *frame, unsigned char *blurred_frame, float *gaussianFunction, unsigned int DIM, 
                        int kernel_size, int rows, int columns, int channels, int *dataTransferTime, int *computationTime){

    unsigned char *device_frame, *device_blurred_frame;
    float *device_gaussianFunction;

    unsigned int size = DIM * sizeof(unsigned char);
    int gaussianSize = kernel_size * sizeof(float);

    time_t startTransferTime;
    time(&startTransferTime);

    cudaError_t error = cudaGetLastError();
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
    //Streams initialization
    int n_streams = 10;
    std::vector<cudaStream_t> streams(n_streams);
    for(int i=0;i<n_streams;i++){
        cudaError_t error1 = cudaStreamCreate(&streams[i]);
    }

    error = cudaGetLastError(); //Error memory allocation

    cudaDeviceProp prop = cudaDeviceProp();
    

    int frameSize = channels * rows * columns;
    int threadXblock = 1024;
    int blocksPerGrid = (DIM/n_streams + threadXblock - 1) / threadXblock;

    cudaEvent_t startComputationTime,stopComputationTime;
    cudaEventCreate(&startComputationTime);
    cudaEventCreate(&stopComputationTime);

    cudaEventRecord(startComputationTime,0);
    for(int i=0;i<n_streams;i++){
        
        int offset = i * frameSize;
        blurframe <<<blocksPerGrid,threadXblock,0,streams[i]>>>(device_frame + offset,
                                                                device_blurred_frame + offset,
                                                                device_gaussianFunction,
                                                                kernel_size,
                                                                rows,
                                                                columns,
                                                                channels);

        error = cudaGetLastError();
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stopComputationTime,0);

    error = cudaGetLastError(); //Error memory allocation

    //destroy streams
    for (int i = 0; i < n_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    error = cudaGetLastError();

    cudaDeviceSynchronize();

    time(&startTransferTime);

    cudaMemcpy(blurred_frame,device_blurred_frame,size,cudaMemcpyDeviceToHost);
    
    time(&stopTransferTime);

    int secondTime = difftime(stopTransferTime,startTransferTime);
    *dataTransferTime = firstTime + secondTime;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,startComputationTime,stopComputationTime);
    *computationTime = elapsedTime;

    cudaDeviceSynchronize();







}








void testGPU()
{
    printf("Hello World from CPU!\n");
    helloFromGPU <<<1, 10>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceReset();
}