#pragma once
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <climits>

using namespace std;

class Video{
    private:
        int width;
        int height;
        int channels;
        int frames;
        unsigned char *data;
        unsigned char **dataByFrame;
        std::vector<unsigned char *> dataVector;
        

        //function used to have consistent vulues of data, dataByFrame, dataVector
        void setDataGivingVector(std::vector<unsigned char *> dataVector){
            dataByFrame = (unsigned char **)malloc(frames * sizeof(unsigned char *));
            data = (unsigned char*)malloc(getDataLenght() * sizeof(unsigned char));
            double datalenght = getDataLenght();
            unsigned int i=0;
            for(unsigned char* frame:dataVector){
                dataByFrame[i] = frame;
                unsigned int offset = i * getFrameSize();
                for(int j=0; j< getFrameSize(); j++){
                    data[offset + j] = frame[j];
                }
                i++;
            }
        }

        //function used to have consistent vulues of data, dataByFrame, dataVector
        void alignData(unsigned char *data){
            this->dataByFrame = (unsigned char **)malloc(frames * sizeof(unsigned char *));
            
            for(unsigned int i=0;i<frames;i++){
                this->dataByFrame[i] = (unsigned char *)malloc(getFrameSize() * sizeof(unsigned char ));
                unsigned int offset = i * getFrameSize();
                for(int j=0; j< getFrameSize(); j++){
                    this->dataByFrame[i][j] = data[offset + j];
                }
                this->dataVector.push_back(this->dataByFrame[i]);
            }
        }

        //function used to have consistent vulues of data, dataByFrame, dataVector
        void alignData(unsigned char ** dataByFrame){
            data = (unsigned char*)malloc(getDataLenght() * sizeof(unsigned char));
            for(int i=0;i<frames;i++){
                unsigned char *dataByFrameTMP = (unsigned char *)malloc(frames * sizeof(unsigned char ));
                for(int j=0; j< width*height*channels; j++){
                    unsigned char value = dataByFrame[i][j];
                    dataByFrameTMP[j] = value;
                    data[i*frames + j] = value;
                }
                dataVector.push_back(dataByFrameTMP);
            }
        }

    public:

        Video(int width, int height, int channels, int frames, std::vector<unsigned char *> dataVector){
            this->width = width;
            this->height = height;
            this->channels = channels;
            this->frames = frames;
            this->dataVector = dataVector;
            setDataGivingVector(dataVector);
        }

        Video(int width, int height, int channels, int frames, unsigned char * data){
            this->width = width;
            this->height = height;
            this->channels = channels;
            this->frames = frames;
            this->data = data;
            alignData(data);
        }

        Video(int width, int height, int channels, int frames, unsigned char ** dataByFrame){
            this->width = width;
            this->height = height;
            this->channels = channels;
            this->frames = frames;
            this->dataByFrame = dataByFrame;
            alignData(dataByFrame);
        }

        int getWidth();
        int getHeight();
        int getChannels();
        int getFrames();
        unsigned int getDataLenght();
        unsigned char* getData();
        int getFrameSize();

        void alignData();

        unsigned char* getDataAtFrame(int frame);
};