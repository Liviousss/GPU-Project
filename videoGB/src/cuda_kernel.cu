#include "../header/cuda_kernel.cuh"

/**
 * GPU matrix for the Gaussian Blur filter applied on video.
 * @param video base image in bytes.
 * @param blurred_video the blurred image in bytes returned by the kernel.
 * @param gaussianMatrix the gaussian matrix in 1D format.
 * @param kernel_size the gaussian matrix kernel size.
 * @param frameRows the image height.
 * @param frameColumns the image width.
 * @param frameChannels the image channels.
 * @param videoFrames the video frames.
*/
__global__ void blurVideo(unsigned char *video,unsigned char *blurred_video,float *gaussianMatrix,int kernel_size, 
                          int frameRows, int frameColumns, int frameChannels, int videoFrames){
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int frameSize = frameChannels*frameRows*frameColumns;

    int frame = idx / frameSize;
    int framePosition = idx % frameSize;
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

            unsigned char pixelValue = video[neighborIdx];
            float gaussianValue = gaussianMatrix[(m + half_kernel_size) * kernel_size + (n + half_kernel_size)];
            blurred_value += pixelValue * gaussianValue;
        }
    }
    
    int intValue = (int)(blurred_value);
    unsigned char unsignedCharValue = (unsigned char)intValue;
    blurred_video[idx] = unsignedCharValue;
    
}

/**
 * GPU matrix for the Gaussian Blur filter applied on video, using the shared memory.
 * @param video base image in bytes.
 * @param blurred_video the blurred image in bytes returned by the kernel.
 * @param gaussianMatrix the gaussian matrix in 1D format.
 * @param kernel_size the gaussian matrix kernel size.
 * @param frameRows the image height.
 * @param frameColumns the image width.
 * @param frameChannels the image channels.
 * @param videoFrames the video frames.
*/
__global__ void blurVideoWithSharedMemory(unsigned char *video,unsigned char *blurred_video,float *gaussianMatrix,int kernel_size, 
                          int frameRows, int frameColumns, int frameChannels, int videoFrames){
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int frameSize = frameChannels*frameRows*frameColumns;

    int frame = idx / frameSize;
    int framePosition = idx % frameSize;
    int row = framePosition / (frameColumns * frameChannels);
    int column = (framePosition % (frameColumns * frameChannels)) / frameChannels;
    int channel = framePosition % frameChannels;

    extern __shared__ unsigned char sharedMem[];

    float value = 0.0f;
    int half_kernel_size = kernel_size/2;
    int DIM = frameChannels*frameRows*frameColumns*videoFrames;

    if(idx>=DIM) return;

    sharedMem[idx] = video[idx];
    __syncthreads();


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

            unsigned char pixelValue = sharedMem[neighborIdx];
            float gaussianValue = gaussianMatrix[(m + half_kernel_size) * kernel_size + (n + half_kernel_size)];
            blurred_value += pixelValue * gaussianValue;
        }
    }
    
    int intValue = (int)(blurred_value);
    unsigned char unsignedCharValue = (unsigned char)intValue;
    blurred_video[idx] = unsignedCharValue;
    
}

/**
 * GPU matrix for the Gaussian Blur filter applied on image.
 * @param image base image in bytes.
 * @param blurred_image the blurred image in bytes returned by the kernel.
 * @param gaussianMatrix the gaussian matrix in 1D format.
 * @param kernel_size the gaussian matrix kernel size.
 * @param height the image height.
 * @param width the image width.
 * @param channels the image channels.
*/
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

void kernel(unsigned char *video, 
            unsigned char* blurred_video, 
            float *gaussianMatrix, 
            unsigned int DIM, 
            int kernel_size, 
            int rows, 
            int columns, 
            int channels,
            int frames,
            int *dataTransferTime,
            int *computationTime){

    unsigned char *device_video, *device_blurred_video;
    float *device_gaussianmatrix;

    int size = DIM * sizeof(unsigned char);
    int gaussianSize = kernel_size * sizeof(float);

    std::chrono::high_resolution_clock::time_point startTransferTime, stopTransferTime;
    startTransferTime = std::chrono::high_resolution_clock::now();

    //VIDEO

    cudaMalloc((void **)&device_video,size);
    cudaMemcpy(device_video,video,size,cudaMemcpyHostToDevice);

    cudaMalloc((void **)&device_blurred_video,size);

    //GAUSSIAN matrix

    cudaMalloc((void **)&device_gaussianmatrix,kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(device_gaussianmatrix,gaussianMatrix,kernel_size * kernel_size * sizeof(float),cudaMemcpyHostToDevice);


    stopTransferTime = std::chrono::high_resolution_clock::now();

    std::chrono::milliseconds firstTransferTime = std::chrono::duration_cast<std::chrono::milliseconds>(stopTransferTime-startTransferTime);
     
    //GPU CODE

    int threadXblock = 1024;
    int blocks = (DIM + threadXblock - 1) / threadXblock;

    cudaEvent_t startComputationTime,stopComputationTime;
    cudaEventCreate(&startComputationTime);
    cudaEventCreate(&stopComputationTime);

    cudaEventRecord(startComputationTime,0);
    blurVideo <<<blocks,threadXblock>>>(device_video,device_blurred_video,device_gaussianmatrix,kernel_size,rows,columns,channels,frames);
    cudaEventRecord(stopComputationTime,0);

    cudaError_t error = cudaGetLastError();

    cudaDeviceSynchronize();

    startTransferTime = std::chrono::high_resolution_clock::now();

    cudaMemcpy(blurred_video,device_blurred_video,size,cudaMemcpyDeviceToHost);
    
    stopTransferTime = std::chrono::high_resolution_clock::now();

    std::chrono::milliseconds secondTransferTime = std::chrono::duration_cast<std::chrono::milliseconds>(stopTransferTime-startTransferTime);
    *dataTransferTime = static_cast<int>(firstTransferTime.count() + secondTransferTime.count());
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,startComputationTime,stopComputationTime);
    *computationTime = elapsedTime;

    //MEMORY FREE
    cudaFree(device_blurred_video);
    cudaFree(device_gaussianmatrix);
    cudaFree(device_video);

    cudaDeviceSynchronize();
};

void kernelUsingStreams(unsigned char *video, unsigned char *blurred_video, float *gaussianMatrix, unsigned int DIM, 
                        int kernel_size, int rows, int columns, int channels, int frames, int *totalTime){

    unsigned char *device_video, *device_blurred_video;
    float *device_gaussianmatrix;

    unsigned int size = DIM * sizeof(unsigned char);
    int gaussianSize = kernel_size * sizeof(float);

    std::chrono::high_resolution_clock::time_point startTransferTime, stopTransferTime;
    startTransferTime = std::chrono::high_resolution_clock::now();

    //VIDEO

    cudaMalloc((void **)&device_video,size);
    cudaMalloc((void **)&device_blurred_video,size);

    //GAUSSIAN matrix

    cudaMalloc((void **)&device_gaussianmatrix,kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(device_gaussianmatrix,gaussianMatrix,kernel_size * kernel_size * sizeof(float),cudaMemcpyHostToDevice);

    stopTransferTime = std::chrono::high_resolution_clock::now();

    std::chrono::milliseconds firstTransferTime = std::chrono::duration_cast<std::chrono::milliseconds>(stopTransferTime-startTransferTime);
     
    //GPU CODE    
    //Streams initialization
    int n_streams = 16;
    std::vector<cudaStream_t> streams(n_streams);
    for(int i=0;i<n_streams;i++){
        cudaError_t error1 = cudaStreamCreate(&streams[i]);
    }

    int streamSize = DIM / n_streams;
    int threadXblock = 1024;
    int blocks = (DIM/frames + threadXblock - 1) / threadXblock;

    cudaEvent_t startComputationTime,stopComputationTime;
    cudaEventCreate(&startComputationTime);
    cudaEventCreate(&stopComputationTime);

    cudaEventRecord(startComputationTime);
    for(int i=0;i<n_streams;i++){
        
        int offset = i * streamSize;
        
        cudaMemcpyAsync(device_video+offset,video+offset,streamSize*sizeof(unsigned char),cudaMemcpyHostToDevice,streams[i]);
        blurVideo <<<blocks,threadXblock,0,streams[i]>>>(device_video + offset,
                                                                device_blurred_video + offset,
                                                                device_gaussianmatrix,
                                                                kernel_size,
                                                                rows,
                                                                columns,
                                                                channels,
                                                                frames);
        cudaMemcpyAsync(blurred_video+offset,device_blurred_video+offset,streamSize*sizeof(unsigned char),cudaMemcpyDeviceToHost,streams[i]);

        
    }

    cudaEventRecord(stopComputationTime);
    for (int i = 0; i < n_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    cudaEventSynchronize(stopComputationTime);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,startComputationTime,stopComputationTime);

    //destroy streams
    for (int i = 0; i < n_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    *totalTime = elapsedTime + firstTransferTime.count();

    //MEMORY FREE
    cudaFree(device_blurred_video);
    cudaFree(device_gaussianmatrix);
    cudaFree(device_video);

    cudaDeviceSynchronize();
}




void kernelUsingSharedMemory(unsigned char *video, 
                             unsigned char *blurred_video, 
                             float *gaussianMatrix, 
                             unsigned int DIM, 
                             int kernel_size, 
                             int rows, 
                             int columns, 
                             int channels, 
                             int frames, 
                             int *dataTransferTime, 
                             int *computationTime){

    unsigned char *device_video, *device_blurred_video;
    float *device_gaussianmatrix;

    int size = DIM * sizeof(unsigned char);
    int gaussianSize = kernel_size * sizeof(float);

    std::chrono::high_resolution_clock::time_point startTransferTime, stopTransferTime;
    startTransferTime = std::chrono::high_resolution_clock::now();

    //VIDEO

    cudaMalloc((void **)&device_video,size);
    cudaMemcpy(device_video,video,size,cudaMemcpyHostToDevice);

    cudaMalloc((void **)&device_blurred_video,size);

    //GAUSSIAN matrix

    cudaMalloc((void **)&device_gaussianmatrix,kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(device_gaussianmatrix,gaussianMatrix,kernel_size * kernel_size * sizeof(float),cudaMemcpyHostToDevice);


    stopTransferTime = std::chrono::high_resolution_clock::now();

    std::chrono::milliseconds firstTransferTime = std::chrono::duration_cast<std::chrono::milliseconds>(stopTransferTime-startTransferTime);
     
    //GPU CODE

    int threadXblock = 1024;
    int blocks = (DIM + threadXblock - 1) / threadXblock;

    cudaEvent_t startComputationTime,stopComputationTime;
    cudaEventCreate(&startComputationTime);
    cudaEventCreate(&stopComputationTime);

    cudaEventRecord(startComputationTime,0);
    blurVideoWithSharedMemory <<<blocks,threadXblock,DIM>>>(device_video,device_blurred_video,device_gaussianmatrix,kernel_size,rows,columns,channels,frames);
    cudaEventRecord(stopComputationTime,0);

    cudaError_t error = cudaGetLastError();

    cudaDeviceSynchronize();

    startTransferTime = std::chrono::high_resolution_clock::now();

    cudaMemcpy(blurred_video,device_blurred_video,size,cudaMemcpyDeviceToHost);
    
    stopTransferTime = std::chrono::high_resolution_clock::now();

    std::chrono::milliseconds secondTransferTime = std::chrono::duration_cast<std::chrono::milliseconds>(stopTransferTime-startTransferTime);
    *dataTransferTime = static_cast<int>(firstTransferTime.count() + secondTransferTime.count());
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,startComputationTime,stopComputationTime);
    *computationTime = elapsedTime;

    //MEMORY FREE
    cudaFree(device_blurred_video);
    cudaFree(device_gaussianmatrix);
    cudaFree(device_video);

    cudaDeviceSynchronize();


}