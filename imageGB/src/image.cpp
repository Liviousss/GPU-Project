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

    int zero_values = 0;
    for(int i=0; i<width * height * channels; i++)
        if(image_data[i]==0)
            zero_values++;

    std::cout<< "zeros " << zero_values <<std::endl;
    
    Image image = Image(width,height,channels,image_data);

    return image;
}

void Image::writeImage(Image image, char *filepath)
{

    int result = stbi_write_jpg(filepath,image.width,image.height,image.channels,image.data,image.width*image.channels);
    if (result==0){
        std::cout << "null ptr";
    }
    else{
        std::cout << "ok";
    }
}


Image Image::createEmptyImage(int width, int height, int channels)
{
    int dim = width * height * channels;
    unsigned char *data = (unsigned char *)malloc(dim * sizeof(unsigned char));

    for(int i=0; i<dim ; i++){
        data[i] = 0;
    }

    Image emptyImage = Image(width,height,channels,data);

    return emptyImage;
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
    return this->channels * this->width * this->height;
}
unsigned char *Image::getData()
{
    return this->data;
}
unsigned char Image::getValueAt(int column, int row)
{
    
    int y = row * width * channels;
    int x = column * channels;

    return this->data[y+x];
}
unsigned char Image::getPosition(int row, int column)
{
    int y = row * width * channels;
    int x = column * channels;

    return x+y;
}