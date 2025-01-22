#include "../header/gaussian_blur.h"
#include <time.h>

float gaussianFunction(int x, int y, int std_dev){

    float std_dev_square = pow(std_dev,2);
    float coeff = 1 / (2 * M_PI * std_dev_square);
    float exp = - (x*x + y*y) / (2 * std_dev_square);

    float result = coeff * pow(M_E,exp);
    return result;
}


void GaussianBlur::generateGaussianMatrix(){
    float ** gaussianMatrix = (float **)malloc(kernel_size * sizeof(float *));
    float sum = 0.0f;

    for(int i=0; i<kernel_size; i++){
        gaussianMatrix[i] = (float *)malloc(kernel_size * sizeof(float));
        for(int j=0; j<kernel_size; j++){
            float value = gaussianFunction(i,j,std_dev);
            gaussianMatrix[i][j] = value;
            sum += value;
        }
    }

    //normalizzazione
    for(int i=0; i<kernel_size; i++){
        for(int j=0; j<kernel_size; j++){
            gaussianMatrix[i][j] /= sum;
        }
    }


    this->gaussianMatrix = gaussianMatrix;
};


/*
CODE FOR THE CPU IMPLEMENTATION : NOT WORKING


Video GaussianBlur::blurVideo(Video video, int* duration)
{
    
    int rows = video.getHeight();
    int columns = video.getWidth();
    int channels = video.getChannels();
    int frames = video.getFrames();
    int frameSize = video.getFrameSize();

    unsigned char *blurredVideoData = (unsigned char*)malloc(video.getDataLenght() * sizeof(unsigned char));

    time_t start = 0;
    time(&start);

    for(int frame=0;frame,frames;frame++){

        for(int i=0; i<rows;i++){
            for(int j=0; j<columns;j++){
                for(int c=0;c<channels;c++){
                    float blurred_value = 0.0;

                    for(int m=-this->half_kernel_size; m<=this->half_kernel_size;m++){
                        for(int n=-this->half_kernel_size; n<=this->half_kernel_size;n++){
                            if ( ! (i+m >= rows || i+m < 0 || j+n+c>=columns || j+n < 0)){
                                unsigned char value = (video.getDataAtFrame(frame))[((i+m)*columns + (j+n)) * channels + c];
                                float gaussianValue = this->gaussianMatrix[m+half_kernel_size][n+half_kernel_size];
                                float valueXblur = value * gaussianValue;
                                blurred_value += valueXblur;
                            }
                                
                        }
                    }
                    
                    int intValue = static_cast<int>(blurred_value);
                    unsigned char unsignedCharValue = static_cast<unsigned char>(intValue);
                    int position = (i*columns + j) * channels + c;
                    blurredVideoData[(frame * frameSize) + position] = unsignedCharValue;
                }

                
                
            }
        }
    }

    time_t stop = 0;
    time(&stop);

    Video blurredVideo = Video(video.getWidth(), video.getHeight(), video.getChannels(), video.getFrames(), blurredVideoData);

    int timeElapsed = difftime(stop,start);
    *duration = timeElapsed;

    return blurredVideo;
}
*/

Video GaussianBlur::blurVideoGPU(Video video, int* dataTransferTime,int* computationTime)
{
    int DIM = video.getDataLenght();
    unsigned char *blurredVideoData = (unsigned char *)malloc(video.getDataLenght() * sizeof(unsigned char));
    float *gaussianKernel = (float *)malloc(kernel_size*kernel_size*sizeof(float));

    for(int i=0;i<kernel_size*kernel_size;i++){
        gaussianKernel[i] = gaussianMatrix[i/kernel_size][i%kernel_size];
    }

    kernel(video.getData(),
            blurredVideoData,
            gaussianKernel,
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
