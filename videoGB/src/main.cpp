#include <opencv2/opencv.hpp>
#include <iostream>
#include "../header/video.h"
#include "../header/gaussian_blur.h"

int main() {
    // Open a video file or capture device (0 for webcam)
    cv::VideoCapture cap("./videos/720p_video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file!" << std::endl;
        return -1;
    }

    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "video width " << frameWidth << std::endl;
    std::cout << "video height " << frameHeight << std::endl;

    std::vector<unsigned char*> frames;
    int channels = 3; // Assume RGB
    int frameSize = frameWidth * frameHeight * channels;
    cv::Mat frame;
    while (cap.read(frame)) {
        // Ensure the frame is in the desired format (e.g., RGB)
        // cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        // Allocate memory for the frame
        unsigned char* frameData = new unsigned char[frameSize];

        // Copy frame data into the allocated memory
        std::memcpy(frameData, frame.data, frameSize);

        // Store the pointer
        frames.push_back(frameData);
    }

    Video video = Video(frameWidth,frameHeight,channels,frames.size(),frames);
    GaussianBlur GB = GaussianBlur();

    int dataTransferTime, computationTime;
    Video blurredVideo = GB.blurVideoGPU(video,&dataTransferTime,&computationTime);
    cv::VideoWriter writer("./videos/720_blurred_video.mp4",
                    cv::VideoWriter::fourcc('m','p','4','v'),
                    fps,
                    cv::Size(frameWidth,frameHeight));
    
    int i=0;
    while(i<blurredVideo.getFrames()){
        unsigned char *data = blurredVideo.getDataAtFrame(i);
        frame = cv::Mat(blurredVideo.getHeight(),blurredVideo.getWidth(),CV_8UC3,data);
        //cv::Mat bgrFrame;
        //cv::cvtColor(frame,bgrFrame,cv::COLOR_RGB2BGR);
        
        //std::memcpy(frame.data,blurredVideo.dataVector[i],blurredVideo.getFrameSize());
        writer.write(frame);
        i++;
    };

    // cv::VideoWriter writerCPU("./videos/720_blurred_video_CPU.mp4",
    //                 cv::VideoWriter::fourcc('m','p','4','v'),
    //                 fps,
    //                 cv::Size(frameWidth,frameHeight));

    // int computationTimeCPU;
    // Video blurredVideoCPU = GB.blurVideo(video,&computationTimeCPU);
    // int j=0;
    // while(j<blurredVideoCPU.getFrames()){
    //     unsigned char *data = blurredVideoCPU.getDataAtFrame(j);
    //     frame = cv::Mat(blurredVideoCPU.getHeight(),blurredVideoCPU.getWidth(),CV_8UC3,data);
        
    //     //std::memcpy(frame.data,blurredVideo.dataVector[i],blurredVideo.getFrameSize());
    //     writerCPU.write(frame);
    //     j++;
    // };


    // Release resources
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
