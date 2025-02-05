#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include "../header/gaussian_blur.h"




int main(int argc, char* argv[]) {

    int user_value;
    if(argc==2){
        try{
            user_value = atoi(argv[1]);
        }catch(std::exception &error){
            std::cerr << "Error: wrong input!: "<<  std::endl;
            return -1;
        }
    }
    
    //Open the webcam
    cv::VideoCapture cap(user_value ? user_value : 0, cv::CAP_V4L2);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open webcam!: "<<  std::endl;
        return -1;
    }

    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int channels = 3;
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::vector<unsigned char*> frames;
    int frameSize = frameWidth * frameHeight * channels;

    GaussianBlur GB = GaussianBlur();

    //Read the video
    cv::Mat frame;
    while (true) {
        cap >> frame;
        
        unsigned char* frameData = new unsigned char[frameSize];

        std::memcpy(frameData, frame.data, frameSize);

        unsigned char* blurredFrame = GB.blurFrame(frameData,frameWidth,frameHeight,channels);

        cv::Mat blurredFramePlayer;
        blurredFramePlayer = cv::Mat(frameHeight,frameWidth,CV_8UC3,blurredFrame);

        cv::imshow("Normal webcam",frame);

        cv::imshow("Blurred Webcam",blurredFramePlayer);
        if (cv::waitKey(1) == 27) // Exit if 'ESC' is pressed
            break;
    }
    return 0;
}
