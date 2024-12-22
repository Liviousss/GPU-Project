#include "../header/image.h"
#include "../header/gaussian_blur.h"
#include <iostream>


void imageAnalisys4k(void);
void ImageAnalisys720(void);

int main(){
    ImageAnalisys720();
    
    imageAnalisys4k();

}

void imageAnalisys4k(void)
{   
    char filepath[] = "./images/4k_image.jpg";

    Image image = Image::loadImage(filepath);

    GaussianBlur GB = GaussianBlur();

    Image blurred_image = GB.blurImage(image);
    char outputFilePath[] = "./images/4k_blurred_image_CPU_prova.jpg";
    Image::writeImage(blurred_image,outputFilePath);


    Image blurred_image_GPU = GB.blurImageGPU(image);
    char outputFilePathGPU[] = "./images/4k_blurred_image_GPU.jpg";
    Image::writeImage(blurred_image_GPU,outputFilePathGPU);

}

void ImageAnalisys720(void)
{
    char filepath[] = "./images/720p_image.jpg";

    Image image = Image::loadImage(filepath);

    GaussianBlur GB = GaussianBlur();

    Image blurred_image = GB.blurImage(image);
    char outputFilePath[] = "./images/720p_image_blurred.jpg";
    Image::writeImage(blurred_image,outputFilePath);

    Image blurred_image_GPU = GB.blurImageGPU(image);
    char outputFilePathGPU[] = "./images/720p_image_blurred_GPU.jpg";
    Image::writeImage(blurred_image_GPU,outputFilePathGPU);

}
