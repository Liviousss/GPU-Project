#include "../header/image.h"
#include "../header/gaussian_blur.h"
#include <iostream>



void imageAnalisys(char* inputFilePath, char* outputFilePathCPU,char* outputFilePathGPU);

int main(){
    
    imageAnalisys("./images/720p_image.jpg","./images/720p_blurred_image_CPU.jpg","./images/720p_blurred_image_GPU.jpg");
    imageAnalisys("./images/1080p_image.jpg","./images/1080p_blurred_image_CPU.jpg","./images/1080p_blurred_image_GPU.jpg");
    imageAnalisys("./images/2k_image.jpg","./images/2k_blurred_image_CPU.jpg","./images/2k_blurred_image_GPU.jpg");
    imageAnalisys("./images/4k_image.jpg","./images/4k_blurred_image_CPU.jpg","./images/4k_blurred_image_GPU.jpg");

}

void imageAnalisys(char* inputFilePath, char* outputFilePathCPU, char* outputFilePathGPU){
    
    printf("-------------------\n");

    Image image = Image::loadImage(inputFilePath);

    GaussianBlur GB = GaussianBlur();

    int durataCPU;
    Image blurred_image = GB.blurImage(image,&durataCPU);
    Image::writeImage(blurred_image,outputFilePathCPU);

    int dataTransferTimeGPU;
    int computationTimeGPU;
    Image blurred_image_GPU = GB.blurImageGPU(image,&dataTransferTimeGPU,&computationTimeGPU);
    Image::writeImage(blurred_image_GPU,outputFilePathGPU);

    printf("Durata CPU = %d\n",durataCPU);
    printf("Durata transferTime GPU = %d\n",dataTransferTimeGPU);
    printf("Durata computationTime GPU = %d millisecondi\n",computationTimeGPU);

    printf("-------------------\n");
}

