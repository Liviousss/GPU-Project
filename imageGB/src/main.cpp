#include "../header/image.h"
#include "../header/gaussian_blur.h"
#include <iostream>



void imageAnalisys(char* title, char* inputFilePath, char* outputFilePathCPU,char* outputFilePathGPU);

int main(){
    
    imageAnalisys("Image resolution: 720p","./images/720p_image.jpg","./images/720p_blurred_image_CPU.jpg","./images/720p_blurred_image_GPU.jpg");
    imageAnalisys("Image resolution: 1080p","./images/1080p_image.jpg","./images/1080p_blurred_image_CPU.jpg","./images/1080p_blurred_image_GPU.jpg");
    imageAnalisys("Image resolution: 2k","./images/2k_image.jpg","./images/2k_blurred_image_CPU.jpg","./images/2k_blurred_image_GPU.jpg");
    imageAnalisys("Image resolution: 4k","./images/4k_image.jpg","./images/4k_blurred_image_CPU.jpg","./images/4k_blurred_image_GPU.jpg");

}

void imageAnalisys(char* title, char* inputFilePath, char* outputFilePathCPU, char* outputFilePathGPU){
    
    printf("-------------------\n");

    std::cout << title << std::endl;

    try{
        //Load the image
        Image image = Image::loadImage(inputFilePath);

        //Create the Gaussian Blur Filter
        GaussianBlur GB = GaussianBlur();

        //Blur the image using the CPU
        int durataCPU;
        Image blurred_image = GB.blurImage(image,&durataCPU);

        //Blur the image using the GPU
        int dataTransferTimeGPU;
        int computationTimeGPU;
        Image blurred_image_GPU = GB.blurImageGPU(image,&dataTransferTimeGPU,&computationTimeGPU);

        //Try write the blurred images
        try{
            Image::writeImage(blurred_image,outputFilePathCPU);
            Image::writeImage(blurred_image_GPU,outputFilePathGPU);
        }
        catch(std::exception &e){
            std::cout << "Image not writed correctly" << std::endl;
            return;
        };

        printf("CPU duration= %d milliseconds\n",durataCPU);
        printf("GPU data transfer time duration= %d milliseconds\n",dataTransferTimeGPU);
        printf("GPU computation time duration= %d milliseconds\n",computationTimeGPU);

        printf("-------------------\n");

    }catch(std::exception &exception){
        std::cout << "Image not loaded correctly, wrong input file path: "<< inputFilePath << std::endl;
        return;
    };

    
}

