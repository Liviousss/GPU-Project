#include "../header/image.h"


Image Image::loadImage(char *filepath)
{
    int width, height, channels;
    unsigned char * image_data = stbi_load(filepath,&width,&height,&channels,0);

    //check if the image is loaded correctly
    if (image_data==nullptr){
        throw std::runtime_error("Image path invalid");
    }

    //image load correctly
    std::cout << "Image loaded\n";
    Image image = Image(width,height,channels,image_data);

    return image;
}

void Image::writeImage(Image image, char *filepath)
{

    //write the image
    int result = stbi_write_jpg(filepath,image.width,image.height,image.channels,image.data,image.width*image.channels);
    if (result==0){
        throw std::runtime_error("Image not written correctly");
    }

    std::cout << "Image written\n";
    
}


int Image::getWidth()
{
    return this->width;
}

int Image::getHeight()
{
    return this->height;
}
int Image::getChannels()
{
    return this->channels;
}
int Image::getDataLenght()
{
    return channels * width * height;
}
unsigned char *Image::getData()
{
    return this->data;
}
