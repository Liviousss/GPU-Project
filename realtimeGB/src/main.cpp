#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include "../header/gaussian_blur.h"




int main() {
    
    //Open the webcam
    cv::VideoCapture cap("/dev/video0", cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS,30);


    // cv::VideoCapture cap("video0", cv::CAP_FFMPEG);
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

    GaussianBlur GB = GaussianBlur();

    int computationTime, dataTransferTime;

    //Read the video
    cv::Mat frame;
    while (true) {
        cap >> frame;
        
        unsigned char* frameData = new unsigned char[frameSize];

        std::memcpy(frameData, frame.data, frameSize);

        unsigned char* blurredFrame = GB.blurFrame(frameData,frameWidth,frameHeight,3,&dataTransferTime,&computationTime);

        cv::Mat blurredFramePlayer;
        blurredFramePlayer = cv::Mat(frameHeight,frameWidth,CV_8UC3,blurredFrame);

        cv::imshow("Normal webcam",frame);

        cv::imshow("Blurred Webcam",blurredFramePlayer);
        if (cv::waitKey(1) == 27) // Exit if 'ESC' is pressed
            break;
    }
    return 0;
}
