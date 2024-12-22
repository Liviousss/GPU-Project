#include "../header/gaussian_blur.h"

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


Image GaussianBlur::blurImage(Image image)
{
    
    int rows = image.getHeight();
    int columns = image.getWidth();

    int channels = image.getChannels();

    unsigned char *blurredImageData = (unsigned char*)malloc(image.getDataLenght() * sizeof(unsigned char));

    for(int i=0; i<rows;i++){
        for(int j=0; j<columns;j++){

            for(int c=0;c<channels;c++){
                float blurred_value = 0.0;

                for(int m=-this->half_kernel_size; m<=this->half_kernel_size;m++){
                    for(int n=-this->half_kernel_size; n<=this->half_kernel_size;n++){
                        if ( ! (i+m >= rows || i+m < 0 || j+n+c>=columns || j+n < 0)){
                            //float value = image.getValueAt(j+n,i+m);
                            unsigned char value = (image.getData())[((i+m)*columns + (j+n)) * channels + c];
                            float gaussianValue = this->gaussianMatrix[m+half_kernel_size][n+half_kernel_size];
                            float valueXblur = value * gaussianValue;
                            blurred_value += valueXblur;
                        }
                            
                    }
                }
                
                //int position = image.getPosition(i,j);
                // printf("Blurredvalue as float: %.2f",blurred_value);
                int intValue = static_cast<int>(blurred_value);
                unsigned char unsignedCharValue = static_cast<unsigned char>(intValue);

                //std::cout << "UC value : "<< unsignedCharValue << "suca" << std::endl;
                //int position = (i*columns + j) * channels;
                int position = (i*columns + j) * channels + c;
                blurredImageData[position] = unsignedCharValue;
                //std::cout << "position : "<< position << "suca" << std::endl;
            }

            
             
        }
    }

    // for(int i=0; i< 10000; i++){
    //     std::cout << blurredImageData[i] << std::endl;
    // }

    int zero_values = 0;
    for(int i=0; i<image.getDataLenght(); i++)
        if(blurredImageData[i]==0)
            zero_values++;

    Image blurredImage = Image(image.getWidth(), image.getHeight(), image.getChannels(), blurredImageData);

    return blurredImage;
}

Image GaussianBlur::blurImageGPU(Image image)
{
    int DIM = image.getDataLenght();
    unsigned char *blurredImageData = (unsigned char *)malloc(image.getDataLenght() * sizeof(unsigned char));
    float *gaussianKernel = (float *)malloc(kernel_size*kernel_size*sizeof(float));

    for(int i=0;i<kernel_size*kernel_size;i++){
        gaussianKernel[i] = gaussianMatrix[i/kernel_size][i%kernel_size];
    }

    kernel(image.getData(),
            blurredImageData,
            gaussianKernel,
            image.getDataLenght(),
            this->kernel_size,
            image.getWidth(),
            image.getHeight(),
            image.getChannels());
    
    cudaDeviceSynchronize();

    Image blurredImage = Image(image.getWidth(), image.getHeight(), image.getChannels(), blurredImageData);
    return blurredImage;
}
