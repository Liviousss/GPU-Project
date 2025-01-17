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
    
    Image image = Image::loadImage(inputFilePath);

    GaussianBlur GB = GaussianBlur();

    Image blurred_image = GB.blurImage(image);
    Image::writeImage(blurred_image,outputFilePathCPU);


    Image blurred_image_GPU = GB.blurImageGPU(image);
    Image::writeImage(blurred_image_GPU,outputFilePathGPU);
}

