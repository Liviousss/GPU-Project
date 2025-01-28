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
        /**
         * Load an image and return an Image object.
         * @param filepath image path.
         * @throw runtime exception if the image isn't loaded correctly
        */
        static Image loadImage(char * filepath);

        /**
         * Write an image.
         * @param image Image object.
         * @param filepath output image path.
         * @throw runtime exception if the image isn't writed correctly.
        */
        static void writeImage(Image image, char * filepath);

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
        
};

