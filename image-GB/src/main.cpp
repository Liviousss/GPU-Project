#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../header/image.h"
#include <iostream>


int main(){
    
    char filepath[] = "./images/4k_image.jpg";

    Image image = Image::createImage(filepath);

    int channels = image.getChannels();
    
    std::cout << "channels : " << channels;
    int a = 10;

    image.modifyImage();

    char outputFilePath[] = "./images/4k_image_modified.jpg";
    Image::writeImage(image,outputFilePath);

}