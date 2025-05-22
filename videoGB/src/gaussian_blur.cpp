#include "../header/gaussian_blur.h"

float gaussianFunction(int x, int y, int std_dev){

    float std_dev_square = pow(std_dev,2);
    float coeff = 1 / (2 * M_PI * std_dev_square);
    float exp = - (x*x + y*y) / (2 * std_dev_square);

    float result = coeff * pow(M_E,exp);
    return result;
}


void GaussianBlur::generateGaussianMatrix(){
    float * gaussianMatrix = (float *)malloc(kernel_size * kernel_size * sizeof(float *));
    float sum = 0.0f;

    for(int i=0; i<kernel_size; i++){
        for(int j=0; j<kernel_size; j++){
            float value = gaussianFunction(i,j,std_dev);
            gaussianMatrix[i*kernel_size + j] = value;
            sum += value;
        }
    }

    //normalization
    for(int i=0; i<kernel_size; i++){
        for(int j=0; j<kernel_size; j++){
            gaussianMatrix[i*kernel_size + j] /= sum;
        }
    }


    this->gaussianMatrix = gaussianMatrix;
};

Video GaussianBlur::blurVideoGPU(Video video, int* dataTransferTime,int* computationTime)
{
    int DIM = video.getDataLenght();
    unsigned char *blurredVideoData = (unsigned char *)malloc(video.getDataLenght() * sizeof(unsigned char));
    
    kernel(video.getData(),
            blurredVideoData,
            this->gaussianMatrix,
            video.getDataLenght(),
            this->kernel_size,
            video.getHeight(),
            video.getWidth(),
            video.getChannels(),
            video.getFrames(),
            dataTransferTime,
            computationTime);
    
    cudaDeviceSynchronize();

    Video blurredVideo = Video(video.getWidth(), video.getHeight(), video.getChannels(), video.getFrames(), blurredVideoData);
    return blurredVideo;
}

Video GaussianBlur::blurVideoGPUusingStreams(Video video, int *totalTime){
    int DIM = video.getDataLenght();
    unsigned char *blurredVideoData = (unsigned char *)malloc(video.getDataLenght() * sizeof(unsigned char));
    
    kernelUsingStreams(video.getData(),
            blurredVideoData,
            this->gaussianMatrix,
            video.getDataLenght(),
            this->kernel_size,
            video.getHeight(),
            video.getWidth(),
            video.getChannels(),
            video.getFrames(),
            totalTime);
    
    cudaDeviceSynchronize();

    Video blurredVideo = Video(video.getWidth(), video.getHeight(), video.getChannels(), video.getFrames(), blurredVideoData);
    return blurredVideo;
}

Video GaussianBlur::blurVideoGPUusingSharedMemory(Video video, int *dataTransferTime, int *computationTime){

    int DIM = video.getDataLenght();
    bool isPossibleToUseSharedMemory = true;
    
    unsigned char *blurredVideoData = (unsigned char *)malloc(video.getDataLenght() * sizeof(unsigned char));
    
    kernelUsingSharedMemory(video.getData(),
            blurredVideoData,
            this->gaussianMatrix,
            video.getDataLenght(),
            this->kernel_size,
            video.getHeight(),
            video.getWidth(),
            video.getChannels(),
            video.getFrames(),
            dataTransferTime,
            computationTime,
            &isPossibleToUseSharedMemory);
    
    cudaDeviceSynchronize();

    if(isPossibleToUseSharedMemory){
        std::cerr << "Video too big for the shared memory" << std::endl;
        std::vector<unsigned char*> vec(0);
        return Video(0,0,0,0,vec);
    }
    
    Video blurredVideo = Video(video.getWidth(), video.getHeight(), video.getChannels(), video.getFrames(), blurredVideoData);
    return blurredVideo;
}
