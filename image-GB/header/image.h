#include "../header/stb_image.h"
#include "../header/stb_image_write.h"
#include <iostream>
#include <vector>

class Pixel{
    public:
        unsigned char R;
        unsigned char G;
        unsigned char B;

        Pixel(int R, int G, int B){
            this->R=R;
            this->G=G;
            this->B=B;
        }
        
        Pixel createPixel(int R, int G, int B){
            return Pixel(R,G,B);
        }
};

class Image{

    private:
        int width;
        int height;
        int channels;
        unsigned char *data;

        std::vector<Pixel> pixels = {};

        Image(int width, int height,int channels,unsigned char* data){
            this->width = width;
            this->height = height;
            this->channels = channels;
            this->data = data;
    }

    public:
        static Image createImage(char * filepath);
        static void writeImage(Image image, char * filepath);

        int getWidth();
        int getHeight();
        int getChannels();
        int getDataLenght();

        void addPixel(Pixel pixel){
            this->pixels.push_back(pixel);
        }

        void modifyDataFromPixels();

        void modifyImage();

};

