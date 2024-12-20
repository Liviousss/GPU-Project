#pragma once

#include "../header/stb_image.h"
#include "../header/stb_image_write.h"
#include <iostream>
#include <vector>
#include <stdlib.h>





class Image{

    private:
        int width;
        int height;
        int channels;
        unsigned char *data;


    public:
        static Image loadImage(char * filepath);
        static void writeImage(Image image, char * filepath);
        static Image createEmptyImage(int width, int height,int channels);

        Image(int width, int height,int channels,unsigned char* data){
            this->width = width;
            this->height = height;
            this->channels = channels;
            this->data = data;
        }

        int getWidth();
        int getHeight();
        int getChannels();
        int getDataLenght();
        unsigned char *getData();

        unsigned char getValueAt(int row, int column);
        unsigned char getPosition(int row, int column);
        
};

