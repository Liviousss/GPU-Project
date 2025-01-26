#include <opencv2/opencv.hpp>
#include <iostream>
#include "../header/video.h"
#include "../header/gaussian_blur.h"




int main() {
    
    //Open the webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open webcam!: "<<  std::endl;
        return -1;
    }

    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::vector<unsigned char*> frames;
    int channels = 3;
    int frameSize = frameWidth * frameHeight * channels;

    //Read the video
    cv::Mat frame;
    while (cap.read(frame)) {
        
        unsigned char* frameData = new unsigned char[frameSize];

        std::memcpy(frameData, frame.data, frameSize);

        frames.push_back(frameData);
    }

    //Create structures for the gaussian blur
    GaussianBlur GB = GaussianBlur();

    

    return 0;
}
