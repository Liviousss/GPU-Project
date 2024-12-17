#include "../header/image.h"

Image Image::createImage(char *filepath)
{
    int width, height, channels;
    std::cout << "Prima di load";
    unsigned char * image_data = stbi_load(filepath,&width,&height,&channels,0);
    std::cout << "Dopo di load";

    Image image = Image(width,height,channels,image_data);
    int dim = width * height * channels;
    for(int i=0; i<dim; i=i+3){
        image.addPixel(Pixel(image_data[i],image_data[i+1],image_data[i+2]));
    }

    return image;
}
int Image::getWidth()
{
    return this->width;
}

int Image::getHeight()
{
    return this->height;
}
int Image::getDataLenght()
{
    return (this->pixels).size();
}