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

int Video::getDataLenght()
{
    return channels * frames * width * height;
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
