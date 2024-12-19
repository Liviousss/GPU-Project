#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../header/image.h"
#include "../header/gaussian_blur.h"
#include <iostream>


int main(){
    
    char filepath[] = "../images/4k_image.jpg";

    Image image = Image::loadImage(filepath);

    int channels = image.getChannels();
    
    std::cout << "channels : " << channels;
    int a = 10;

    //image.modifyImage();

    

    GaussianBlur GB = GaussianBlur();

    Image blurred_image = GB.blurImage(image);

    char outputFilePath[] = "../images/4k_image_modified_1.jpg";
    Image::writeImage(blurred_image,outputFilePath);
    

}