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

__global__ void blurVideoStreams(unsigned char *video,unsigned char *blurred_video,float *gaussianMatrix,int kernel_size, 
                          int frameRows, int frameColumns, int frameChannels, int videoFrames,int streamSize){
    
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

    if(idx>=streamSize) return;


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

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

__global__ void blurVideoWithSharedMemoryAndStreams(
    unsigned char *video, unsigned char *blurred_video,
    float *gaussianMatrix, int kernel_size, 
    int frameRows, int frameColumns, int frameChannels,
    int frameOffset)
{
    int half_kernel_size = kernel_size / 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * BLOCK_WIDTH + tx;
    int row = blockIdx.y * BLOCK_HEIGHT + ty;
    int channel = blockIdx.z;

    int shared_width = BLOCK_WIDTH + kernel_size;
    int shared_height = BLOCK_HEIGHT + kernel_size;

    extern __shared__ unsigned char sharedMem[];


    for (int m = -half_kernel_size; m <= half_kernel_size; m++) {
        for (int n = -half_kernel_size; n <= half_kernel_size; n++) {
            int shared_x = tx + n + half_kernel_size;
            int shared_y = ty + m + half_kernel_size;

            int neighborRow = row + m;
            int neighborColumn = col + n;

            neighborRow = min(max(neighborRow, 0), frameRows - 1);
            neighborColumn = min(max(neighborColumn, 0), frameColumns - 1);

            int global_idx = ((neighborRow * frameColumns + neighborColumn) * frameChannels + channel) + frameOffset;

            if (shared_x < shared_width && shared_y < shared_height)
                sharedMem[shared_y * shared_width + shared_x] = video[global_idx];
        }
    }

    __syncthreads();

    if (row < frameRows && col < frameColumns) {
        float blured_value = 0.0f;

        for (int m = -half_kernel_size; m <= half_kernel_size; ++m) {
            for (int n = -half_kernel_size; n <= half_kernel_size; ++n) {
                int column = tx + n + half_kernel_size;
                int row = ty + m + half_kernel_size;

                unsigned char pixel = sharedMem[row * shared_width + column];
                float gaussianValue = gaussianMatrix[(m + half_kernel_size) * kernel_size + (n + half_kernel_size)];
                blured_value += pixel * gaussianValue;
            }
        }

        int output_idx = ((row * frameColumns + col) * frameChannels + channel) + frameOffset;
        blurred_video[output_idx] = static_cast<unsigned char>(blured_value);
    }
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

    //streamSize as multiple of imageSize
    int streamSize = DIM / n_streams;
    int imageSize = rows*columns*channels;
    int framesXstream = streamSize/imageSize;
    streamSize = imageSize*framesXstream;
    int threadXblock = 1024;
    int blocks = (DIM/n_streams + threadXblock - 1) / threadXblock;

    cudaEvent_t startComputationTime,stopComputationTime;
    cudaEventCreate(&startComputationTime);
    cudaEventCreate(&stopComputationTime);

    cudaEventRecord(startComputationTime);
    for(int i=0;i<n_streams;i++){
        
        int offset = i * streamSize;
        
        cudaMemcpyAsync(device_video+offset,video+offset,streamSize*sizeof(unsigned char),cudaMemcpyHostToDevice,streams[i]);
        blurVideoStreams <<<blocks,threadXblock,0,streams[i]>>>(device_video + offset,
                                                                device_blurred_video + offset,
                                                                device_gaussianmatrix,
                                                                kernel_size,
                                                                rows,
                                                                columns,
                                                                channels,
                                                                frames,
                                                                streamSize);
        cudaMemcpyAsync(blurred_video+offset,device_blurred_video+offset,streamSize*sizeof(unsigned char),cudaMemcpyDeviceToHost,streams[i]);

        
    }

    cudaEventRecord(stopComputationTime);
    for (int i = 0; i < n_streams; i++) {
        cudaError_t error = cudaStreamSynchronize(streams[i]);
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
                             int *computationTime,
                             bool *isPossibleToUseSharedMemory){

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

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if(blocks*threadXblock>deviceProp.sharedMemPerBlock){
        *isPossibleToUseSharedMemory = false;
        return ;
    }

    cudaEventRecord(startComputationTime,0);
    blurVideoWithSharedMemory <<<blocks,threadXblock,blocks*threadXblock>>>(device_video,device_blurred_video,device_gaussianmatrix,kernel_size,rows,columns,channels,frames);
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


void kernelUsingSharedMemoryAndStreams(unsigned char *video, 
    unsigned char *blurred_video, 
    float *gaussianMatrix, 
    unsigned int DIM, 
    int kernel_size, 
    int rows, 
    int columns, 
    int channels, 
    int frames,
    int *totalTime)
{

    unsigned char *device_video, *device_blurred_video;
    float *device_gaussianmatrix;

    unsigned int size = DIM * sizeof(unsigned char);
    int gaussianSize = kernel_size * kernel_size * sizeof(float);

    std::chrono::high_resolution_clock::time_point startTransferTime, stopTransferTime;

    startTransferTime = std::chrono::high_resolution_clock::now();

    cudaMalloc((void **)&device_video,size);
    cudaMalloc((void **)&device_blurred_video,size);

    //GAUSSIAN matrix

    cudaMalloc((void **)&device_gaussianmatrix,kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(device_gaussianmatrix,gaussianMatrix,kernel_size * kernel_size * sizeof(float),cudaMemcpyHostToDevice);

    stopTransferTime = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds firstTransferTime = std::chrono::duration_cast<std::chrono::milliseconds>(stopTransferTime - startTransferTime);

    int frameSize = rows * columns * channels;
    int n_streams = frames;

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++)
        cudaStreamCreate(&streams[i]);

    dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 gridDim((columns + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
                 (rows + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT,
                 channels);

    size_t sharedMemorySize = (BLOCK_WIDTH + 2 * (kernel_size / 2)) *
                           (BLOCK_HEIGHT + 2 * (kernel_size / 2)) * sizeof(unsigned char);

    cudaEvent_t startComputationTime, stopComputationTime;
    cudaEventCreate(&startComputationTime);
    cudaEventCreate(&stopComputationTime);
    cudaEventRecord(startComputationTime);

    for (int i = 0; i < n_streams; i++) {
        int offset = i * frameSize;

        cudaMemcpyAsync(device_video + offset, video + offset, frameSize * sizeof(unsigned char), cudaMemcpyHostToDevice, streams[i]);

        blurVideoWithSharedMemoryAndStreams<<<gridDim, blockDim, sharedMemorySize, streams[i]>>>(
            device_video, device_blurred_video, device_gaussianmatrix, kernel_size, rows, columns, channels, offset);

        cudaMemcpyAsync(blurred_video + offset, device_blurred_video + offset, frameSize * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaEventRecord(stopComputationTime);
    for (int i = 0; i < n_streams; i++)
        cudaStreamSynchronize(streams[i]);
    cudaEventSynchronize(stopComputationTime);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startComputationTime, stopComputationTime);

    for (int i = 0; i < n_streams; i++)
        cudaStreamDestroy(streams[i]);

    *totalTime = static_cast<int>(elapsedTime + firstTransferTime.count());

    cudaFree(device_video);
    cudaFree(device_blurred_video);
    cudaFree(device_gaussianmatrix);

}