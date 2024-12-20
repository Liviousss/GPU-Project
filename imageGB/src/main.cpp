#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../header/image.h"
#include "../header/gaussian_blur.h"
#include <iostream>


void imageAnalisys4k(void);
void ImageAnalisys720(void);

int main(){
    ImageAnalisys720();
    


}

void imageAnalisys4k(void)
{   
    char filepath[] = "../images/4k_image.jpg";

    Image image = Image::loadImage(filepath);

    GaussianBlur GB = GaussianBlur();

    Image blurred_image = GB.blurImage(image);

    char outputFilePath[] = "../images/4k_basic_struct_image_copy.jpg";
    Image::writeImage(blurred_image,outputFilePath);

}

void ImageAnalisys720(void)
{
    char filepath[] = "../images/720p_image.jpg";

    Image image = Image::loadImage(filepath);

    GaussianBlur GB = GaussianBlur();

    Image blurred_image = GB.blurImage(image);

    char outputFilePath[] = "../images/720p_image_blurred.jpg";
    Image::writeImage(blurred_image,outputFilePath);

}
