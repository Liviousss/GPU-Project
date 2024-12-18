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
void Image::writeImage(Image image, char *filepath)
{
    image.modifyDataFromPixels();
    stbi_write_jpg(filepath,image.width,image.height,image.channels,image.data,image.width*image.channels);
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
    return (this->pixels).size();
}
void Image::modifyDataFromPixels()
{
    int dim = this->width * this->height * this->channels;
    for(int i=0,j=0;i<dim;i=i+3,j=j+1){
        // this->data[i] = this->pixels[j].R;
        // this->data[i+1] = this->pixels[j].G;
        // this->data[i+2] = this->pixels[j].B;

        this->data[i] = 230;
        this->data[i+1] = 230;
        this->data[i+2] = 230;
    }
}
void Image::modifyImage()
{

    for(Pixel p : this->pixels){
        p.R=50;
        p.G=50;
        p.B=50;
    }
}