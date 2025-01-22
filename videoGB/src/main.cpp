#include <opencv2/opencv.hpp>
#include <iostream>
#include "../header/video.h"
#include "../header/gaussian_blur.h"


int BlurVideo(std::string src,std::string dest);


int main() {
    
    std::string video720p_src = "./videos/720p_video.mp4";
    std::string video720p_dest = "./videos/720p_blurred_video.mp4";

    BlurVideo(video720p_src,video720p_dest);
    
    return 0;
}


int BlurVideo(std::string src,std::string dest){
    //Open a video
    cv::VideoCapture cap(src);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open src video! Path: "<< src << std::endl;
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
    Video video = Video(frameWidth,frameHeight,channels,frames.size(),frames);
    GaussianBlur GB = GaussianBlur();

    //Use the GPU to blur the video
    int dataTransferTime, computationTime;
    Video blurredVideo = GB.blurVideoGPU(video,&dataTransferTime,&computationTime);

    //Create video writer
    cv::VideoWriter writer(dest,
                    cv::VideoWriter::fourcc('m','p','4','v'),
                    fps,
                    cv::Size(frameWidth,frameHeight));

    if(!writer.isOpened()){
        std::cerr << "Error: Cannot open dest video! Path: "<< dest << std::endl;
        return -1;
    }
    
    //write the video
    int i=0;
    while(i<blurredVideo.getFrames()){
        unsigned char *data = blurredVideo.getDataAtFrame(i);
        frame = cv::Mat(blurredVideo.getHeight(),blurredVideo.getWidth(),CV_8UC3,data);
        writer.write(frame);
        i++;
    };

    // Release resources
    cap.release();
    cv::destroyAllWindows();

    std::cout << "Video blurred correctly" << std::endl;
    printf("GPU data transfer time : %d seconds\n",dataTransferTime);
    printf("GPU computation time : %d milliseconds\n",dataTransferTime);

    return 0;

}


/*
    CODE FOR CPU GAUSSIAN BLUR


    cv::VideoWriter writerCPU("./videos/720_blurred_video_CPU.mp4",
                    cv::VideoWriter::fourcc('m','p','4','v'),
                    fps,
                    cv::Size(frameWidth,frameHeight));

    int computationTimeCPU;
    Video blurredVideoCPU = GB.blurVideo(video,&computationTimeCPU);
    int j=0;
    while(j<blurredVideoCPU.getFrames()){
        unsigned char *data = blurredVideoCPU.getDataAtFrame(j);
        frame = cv::Mat(blurredVideoCPU.getHeight(),blurredVideoCPU.getWidth(),CV_8UC3,data);
        
        //std::memcpy(frame.data,blurredVideo.dataVector[i],blurredVideo.getFrameSize());
        writerCPU.write(frame);
        j++;
    };
*/