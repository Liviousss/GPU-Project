#define STB_IMAGE_IMPLEMENTATION
#include "../header/image.h"
#include <iostream>


int main(){
    
    char filepath[] = "/home/livio/Scrivania/GPU Project/GPU-Project/image-GB/images/4k_image.jpg";

    Image image = Image::createImage(filepath);

    int dim = image.getDataLenght();
    
    std::cout << "dimesione : " << dim;
    int a = 10;

}