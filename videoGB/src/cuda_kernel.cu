#include "../header/cuda_kernel.cuh"

__global__ void helloFromGPU (void) {
    printf("Hello World from Jetson GPU!\n");
}


__global__ void blurVideo(unsigned char *video,unsigned char *blurred_video,float *gaussianMatrix,int kernel_size, 
                          int frameRows, int frameColumns, int frameChannels, int videoFrames){
    
    

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int frameSize = frameChannels*frameRows*frameColumns;

    int frame = idx / frameSize;
    int framePosition = idx % frameSize;
    // int row = framePosition / (frameColumns * frameChannels);
    // int columnOffset = framePosition % (frameColumns * frameChannels);
    // int column = columnOffset / frameColumns;
    // int channel = columnOffset % frameColumns;
    int row = framePosition / (frameColumns * frameChannels);
    int column = (framePosition % (frameColumns * frameChannels)) / frameChannels;
    int channel = framePosition % frameChannels;


    float value = 0.0f;
    int half_kernel_size = kernel_size/2;
    int DIM = frameChannels*frameRows*frameColumns*videoFrames;

    if(idx>=DIM) return;


    float blurred_value = 0.0;

    for(int m=-half_kernel_size; m<=half_kernel_size;m++){
        for(int n=-half_kernel_size; n<=half_kernel_size;n++){
            int neighborRow = row + m;
            int neighborColumn = column + n;

            if (neighborRow < 0 || neighborRow >= frameRows || 
                neighborColumn < 0 || neighborColumn >= frameColumns) {
                continue;
            }

            int neighborIdx = ((frame * frameRows + neighborRow) * frameColumns + neighborColumn) * frameChannels + channel;

            // Access pixel value and Gaussian weight
            unsigned char pixelValue = video[neighborIdx];
            float gaussianValue = gaussianMatrix[(m + half_kernel_size) * kernel_size + (n + half_kernel_size)];

            // Accumulate blurred value
            blurred_value += pixelValue * gaussianValue;
        }
    }
    
    int intValue = (int)(blurred_value);
    unsigned char unsignedCharValue = (unsigned char)intValue;
    blurred_video[idx] = unsignedCharValue;
    
}

void kernel(unsigned char *video, 
            unsigned char* blurred_video, 
            float *gaussianFunction, 
            int DIM, 
            int kernel_size, 
            int rows, 
            int columns, 
            int channels,
            int frames,
            int *dataTransferTime,
            int *computationTime){

    unsigned char *device_video, *device_blurred_video;
    float *device_gaussianFunction;

    int size = DIM * sizeof(unsigned char);
    int gaussianSize = kernel_size * sizeof(float);

    time_t startTransferTime;
    time(&startTransferTime);

    //IMAGE

    cudaMalloc((void **)&device_video,size);
    cudaMemcpy(device_video,video,size,cudaMemcpyHostToDevice);

    cudaMalloc((void **)&device_blurred_video,size);

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
    blurVideo <<<blocksPerGrid,threadXblock>>>(device_video,device_blurred_video,device_gaussianFunction,kernel_size,rows,columns,channels,frames);
    cudaEventRecord(stopComputationTime,0);

    cudaDeviceSynchronize();

    time(&startTransferTime);

    cudaMemcpy(blurred_video,device_blurred_video,size,cudaMemcpyDeviceToHost);
    
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