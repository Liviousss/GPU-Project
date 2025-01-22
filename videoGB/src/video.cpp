#include "../header/video.h"

int Video::getWidth()
{
    return width;
}

int Video::getHeight()
{
    return height;
}

int Video::getChannels()
{
    return channels;
}

int Video::getFrames()
{
    return frames;
}

unsigned int Video::getDataLenght()
{
    unsigned int value = static_cast<unsigned int>(channels) * static_cast<unsigned int>(frames) 
                        * static_cast<unsigned int>(width) * static_cast<unsigned int>(height);
    return value;
}

unsigned char *Video::getData()
{
    return data;
}

int Video::getFrameSize()
{
    return width*height*channels;
}
void Video::alignData()
{
    alignData(data);
}

unsigned char *Video::getDataAtFrame(int frame)
{
    return dataVector.at(frame);
}
