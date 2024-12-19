#include "../header/gaussian_blur.h"

float gaussianFunction(int x, int y, int std_dev){
    float result = (1 / (2 * M_PI * (std_dev * std_dev))) * pow(M_E,(- (x*x + y*y) / (2 * (std_dev * std_dev))));
    return result;
}

void GaussianBlur::generateGaussianMatrix()
{

    //std::vector<std::vector<float>> gaussianMatrix;

    for(int i=0; i<this->kernel_size; i++){
        this->gaussianMatrix.push_back(std::vector<float>());
        for(int j=0; j<this->kernel_size; j++){
            float value = gaussianFunction(j,i,this->std_dev);
            this->gaussianMatrix[i].push_back(value);
        }
    }

    //this->gaussianMatrix = gaussianMatrix;
}

std::vector<std::vector<float>> GaussianBlur::getGaussianMatrix(){
    return this->gaussianMatrix;
}

Image GaussianBlur::blurImage(Image image)
{
    
    std::vector<std::vector<Pixel>> imagePixelMatrix = image.getPixelMatrix();

    std::vector<std::vector<Pixel>> blurredImageMatrix = image.getPixelMatrix();

    for(int i=0; i<imagePixelMatrix.size();i++){
        for(int j=0; j<imagePixelMatrix[i].size();j++){
            Pixel p = imagePixelMatrix[i][j];
            Pixel blurredPixel = p;

            for(int m=0; m<=this->half_kernel_size;m++){
                for(int n=0; n<=this->half_kernel_size;n++){
                    if ( ! (i+m >= imagePixelMatrix.size() || j+n>=imagePixelMatrix[i].size())){
                        Pixel imageXgaussian = imagePixelMatrix[i+m][j+n].mul(this->gaussianMatrix[m][n]);
                        blurredPixel = blurredPixel.add(imageXgaussian);
                    }
                        
                }
            }

            blurredImageMatrix[i][j] = blurredPixel;
        }
    }

    Image blurredImage = Image::createEmptyImage(image.getWidth(), image.getHeight(), image.getChannels());
    blurredImage.addPixelMatrix(blurredImageMatrix);

    return blurredImage;
}
