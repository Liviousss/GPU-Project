#pragma once

#include "../header/stb_image.h"
#include "../header/stb_image_write.h"
#include <iostream>
#include <vector>
#include <stdlib.h>



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

        Pixel mul(float offset){
            Pixel p(0,0,0);
            p.R = this->R * offset;
            p.G = this->G * offset;
            p.B = this->B * offset;

            return p;
        }

        Pixel add(Pixel pixel){
            Pixel p(0,0,0);
            p.R = this->R + pixel.R;
            p.G = this->G + pixel.G;
            p.B = this->B + pixel.B;

            return p;
        }
};

class Image{

    private:
        int width;
        int height;
        int channels;
        unsigned char *data;

        std::vector<Pixel> pixels = {};
        std::vector<std::vector<Pixel>> pixelMatrix = {};

        Image(int width, int height,int channels,unsigned char* data){
            this->width = width;
            this->height = height;
            this->channels = channels;
            this->data = data;
        }

        void generatePixelMatrix();

    public:
        static Image loadImage(char * filepath);
        static void writeImage(Image image, char * filepath);
        static Image createEmptyImage(int width, int height,int channels);

        int getWidth();
        int getHeight();
        int getChannels();
        int getDataLenght();
        unsigned char *getData();
        std::vector<std::vector<Pixel>> getPixelMatrix();

        void addPixelMatrix(std::vector<std::vector<Pixel>> pixelMatrix){
            this->pixelMatrix = pixelMatrix;

            this->pixels.clear();
            for(int i=0; i< pixelMatrix.size(); i++){
                for(int j=0; j< pixelMatrix[i].size(); j++){
                    this->pixels.push_back(pixelMatrix[i][j]);
                }
            }

            modifyDataFromPixels();


        }

        void addPixel(Pixel pixel){
            this->pixels.push_back(pixel);
        }

        void modifyDataFromPixels();

        void modifyImage();

};

