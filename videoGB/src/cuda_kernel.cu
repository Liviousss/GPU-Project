#include "../header/cuda_kernel.cuh"

/**
 * GPU function for the Gaussian Blur filter applied on video.
 * @param video base image in bytes.
 * @param blurred_video the blurred image in bytes returned by the kernel.
 * @param gaussianMatrix the gaussian matrix in 1D format.
 * @param kernel_size the gaussian function kernel size.
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
 * GPU function for the Gaussian Blur filter applied on image.
 * @param image base image in bytes.
 * @param blurred_image the blurred image in bytes returned by the kernel.
 * @param gaussianMatrix the gaussian matrix in 1D format.
 * @param kernel_size the gaussian function kernel size.
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
    float *device_gaussianFunction;

    int size = DIM * sizeof(unsigned char);
    int gaussianSize = kernel_size * sizeof(float);

    time_t startTransferTime;
    time(&startTransferTime);

    //VIDEO

    cudaMalloc((void **)&device_video,size);
    cudaMemcpy(device_video,video,size,cudaMemcpyHostToDevice);

    cudaMalloc((void **)&device_blurred_video,size);

    //GAUSSIAN FUNCTION

    cudaMalloc((void **)&device_gaussianFunction,kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(device_gaussianFunction,gaussianMatrix,kernel_size * kernel_size * sizeof(float),cudaMemcpyHostToDevice);


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

    cudaError_t error = cudaGetLastError();

    cudaDeviceSynchronize();

    time(&startTransferTime);

    cudaMemcpy(blurred_video,device_blurred_video,size,cudaMemcpyDeviceToHost);
    
    time(&stopTransferTime);

    int secondTime = difftime(stopTransferTime,startTransferTime);
    *dataTransferTime = firstTime + secondTime;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,startComputationTime,stopComputationTime);
    *computationTime = elapsedTime;

    //MEMORY FREE
    cudaFree(device_blurred_video);
    cudaFree(device_gaussianFunction);
    cudaFree(device_video);

    cudaDeviceSynchronize();
};

void kernelUsingStreams(unsigned char *video, unsigned char *blurred_video, float *gaussianMatrix, unsigned int DIM, 
                        int kernel_size, int rows, int columns, int channels, int frames, int *dataTransferTime, int *computationTime){

    unsigned char *device_video, *device_blurred_video;
    float *device_gaussianFunction;

    unsigned int size = DIM * sizeof(unsigned char);
    int gaussianSize = kernel_size * sizeof(float);

    time_t startTransferTime;
    time(&startTransferTime);

    //VIDEO

    cudaMalloc((void **)&device_video,size);

    cudaMemcpy(device_video,video,size,cudaMemcpyHostToDevice);

    cudaMalloc((void **)&device_blurred_video,size);

    //GAUSSIAN FUNCTION

    cudaMalloc((void **)&device_gaussianFunction,kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(device_gaussianFunction,gaussianMatrix,kernel_size * kernel_size * sizeof(float),cudaMemcpyHostToDevice);

    time_t stopTransferTime;
    time(&stopTransferTime);

    int firstTime = difftime(stopTransferTime,startTransferTime);
     
    //GPU CODE    
    //Streams initialization
    std::vector<cudaStream_t> streams(frames);
    for(int i=0;i<frames;i++){
        cudaError_t error1 = cudaStreamCreate(&streams[i]);
    }

    int frameSize = channels * rows * columns;
    int threadXblock = 1024;
    int blocksPerGrid = (DIM/frames + threadXblock - 1) / threadXblock;

    cudaEvent_t startComputationTime,stopComputationTime;
    cudaEventCreate(&startComputationTime);
    cudaEventCreate(&stopComputationTime);

    cudaEventRecord(startComputationTime,0);
    for(int i=0;i<frames;i++){
        
        int offset = i * frameSize;
        blurImage <<<blocksPerGrid,threadXblock,0,streams[i]>>>(device_video + offset,
                                                                device_blurred_video + offset,
                                                                device_gaussianFunction,
                                                                kernel_size,
                                                                rows,
                                                                columns,
                                                                channels);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stopComputationTime,0);

    //destroy streams
    for (int i = 0; i < frames; i++) {
        cudaStreamDestroy(streams[i]);
    }

    time(&startTransferTime);

    cudaMemcpy(blurred_video,device_blurred_video,size,cudaMemcpyDeviceToHost);
    
    time(&stopTransferTime);

    int secondTime = difftime(stopTransferTime,startTransferTime);
    *dataTransferTime = firstTime + secondTime;
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,startComputationTime,stopComputationTime);
    *computationTime = elapsedTime;

    //MEMORY FREE
    cudaFree(device_blurred_video);
    cudaFree(device_gaussianFunction);
    cudaFree(device_video);

    cudaDeviceSynchronize();







}