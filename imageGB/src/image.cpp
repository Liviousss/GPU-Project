#include "../header/image.h"



Image Image::loadImage(char *filepath)
{
    int width, height, channels;
    unsigned char * image_data = stbi_load(filepath,&width,&height,&channels,0);

    if (image_data==nullptr){
        std::cout << "null ptr";
    }
    else{
        std::cout << "ok";
    }

    Image image = Image(width,height,channels,image_data);
    int dim = width * height * channels;
    for(int i=0; i<dim; i=i+3){
        image.addPixel(Pixel(image_data[i],image_data[i+1],image_data[i+2]));
    }

    image.generatePixelMatrix();

    return image;
}

void Image::writeImage(Image image, char *filepath)
{
    image.modifyDataFromPixels();
    stbi_write_jpg(filepath,image.width,image.height,image.channels,image.data,image.width*image.channels);
}


Image Image::createEmptyImage(int width, int height, int channels)
{
    int dim = width * height * channels;
    unsigned char *data = (unsigned char *)malloc(dim * sizeof(unsigned char));

    for(int i=0; i<dim ; i++){
        data[i] = 0;
    }

    Image emptyImage = Image(width,height,channels,data);

    for(int i=0; i<dim; i=i+3){
        emptyImage.addPixel(Pixel(0,0,0));
    }
    emptyImage.generatePixelMatrix();

    return emptyImage;
}

void Image::generatePixelMatrix(){
    std::vector<std::vector<Pixel>> vec;
    for(int i=0; i<pixels.size(); i++){
        int y = (int)(i / this->height);
        int x = i % this->width;
        
        if(y==vec.size()){
            std::vector<Pixel> row;
            vec.push_back(row);
        }
        vec[y].push_back(this->pixels[i]);
    }

    this->pixelMatrix = vec;
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
unsigned char *Image::getData()
{
    return this->data;
}

std::vector<std::vector<Pixel>> Image::getPixelMatrix(){
    return this->pixelMatrix;
}

void Image::modifyDataFromPixels()
{
    int dim = this->width * this->height * this->channels;
    for(int i=0,j=0;i<dim;i=i+3,j=j+1){
        this->data[i] = this->pixels[j].R;
        this->data[i+1] = this->pixels[j].G;
        this->data[i+2] = this->pixels[j].B;
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