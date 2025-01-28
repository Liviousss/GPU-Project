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


Image GaussianBlur::blurImage(Image image, int* duration)
{
    
    int rows = image.getHeight();
    int columns = image.getWidth();

    int channels = image.getChannels();

    unsigned char *blurredImageData = (unsigned char*)malloc(image.getDataLenght() * sizeof(unsigned char));

    time_t start = 0;
    time(&start);

    for(int i=0; i<rows;i++){
        for(int j=0; j<columns;j++){

            for(int c=0;c<channels;c++){
                float blurred_value = 0.0;

                for(int m=-this->half_kernel_size; m<=this->half_kernel_size;m++){
                    for(int n=-this->half_kernel_size; n<=this->half_kernel_size;n++){
                        if ( ! (i+m >= rows || i+m < 0 || j+n+c>=columns || j+n < 0)){
                            unsigned char value = (image.getData())[((i+m)*columns + (j+n)) * channels + c];
                            int pos = (m+half_kernel_size) * kernel_size + n+half_kernel_size;
                            float gaussianValue = this->gaussianMatrix[pos];
                            float valueXblur = value * gaussianValue;
                            blurred_value += valueXblur;
                        }
                            
                    }
                }
                
                int intValue = static_cast<int>(blurred_value);
                unsigned char unsignedCharValue = static_cast<unsigned char>(intValue);
                int position = (i*columns + j) * channels + c;
                blurredImageData[position] = unsignedCharValue;
            }

            
             
        }
    }

    time_t stop = 0;
    time(&stop);

    Image blurredImage = Image(image.getWidth(), image.getHeight(), image.getChannels(), blurredImageData);

    int timeElapsed = difftime(stop,start);
    *duration = timeElapsed;

    return blurredImage;
}

Image GaussianBlur::blurImageGPU(Image image, int* dataTransferTime,int* computationTime)
{
    int DIM = image.getDataLenght();
    unsigned char *blurredImageData = (unsigned char *)malloc(image.getDataLenght() * sizeof(unsigned char));
    

    kernel(image.getData(),
            blurredImageData,
            gaussianMatrix,
            image.getDataLenght(),
            this->kernel_size,
            image.getHeight(),
            image.getWidth(),
            image.getChannels(),
            dataTransferTime,
            computationTime);
    
    cudaDeviceSynchronize();

    Image blurredImage = Image(image.getWidth(), image.getHeight(), image.getChannels(), blurredImageData);
    return blurredImage;
}
