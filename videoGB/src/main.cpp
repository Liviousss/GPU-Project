#include <opencv2/opencv.hpp>
#include <iostream>
#include "../header/video.h"
#include "../header/gaussian_blur.h"


enum Mode {BASIC,STREAMS,SHARED_MEMORY,SHARED_STREAMS};

int BlurVideo(std::string src,std::string dest, Mode mode);


int main() {

    std::string videosd_src = "./videos/sd_video.mp4";
    std::string videosd_dest = "./videos/sd_blurred_video.mp4";

    BlurVideo(videosd_src,videosd_dest,BASIC);
    
    std::string video720p_src = "./videos/720p_video.mp4";
    std::string video720p_dest = "./videos/720p_blurred_video.mp4";

    BlurVideo(video720p_src,video720p_dest, BASIC);
    
    return 0;
}


int BlurVideo(std::string src,std::string dest, Mode mode){
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
        
    Video blurredVideo(0,0,0,0,std::vector<unsigned char*>(0));
    printf("-------------------\n");
    switch (mode)
    {
    case Mode::BASIC:
        {
            int dataTransferTime = -1, computationTime = -1;
            blurredVideo = GB.blurVideoGPU(video,&dataTransferTime,&computationTime);
            if(dataTransferTime!=-1 && computationTime!=-1){
                printf("GPU data transfer time : %d milliseconds\n",dataTransferTime);
                printf("GPU computation time : %d milliseconds\n",computationTime);
            }
            else{
                printf("Something went wrong with basic video blur function\n");
                return -1;
            }
            break;
        }

    case Mode::STREAMS:
        {
            int totalTimeWithStreams = -1;
            blurredVideo = GB.blurVideoGPUusingStreams(video,&totalTimeWithStreams);
            if(totalTimeWithStreams!=-1){
                printf("GPU total time using streams: %d milliseconds\n",totalTimeWithStreams);
            }
            else{
                printf("Something went wrong with video blur using streams\n");
                return -1;
            }
            break;
        }
        

    case Mode::SHARED_MEMORY:
        {
            int dataTransferTimeWithSharedMem = -1, computationTimeWithSharedMem = -1;
            blurredVideo = GB.blurVideoGPUusingSharedMemory(video,&dataTransferTimeWithSharedMem,&computationTimeWithSharedMem);
            if(dataTransferTimeWithSharedMem!=-1 && computationTimeWithSharedMem!=-1){
                printf("GPU data transfer time using shared memory: %d milliseconds\n",dataTransferTimeWithSharedMem);
                printf("GPU computation time using shared memory: %d milliseconds\n",computationTimeWithSharedMem);
            }
            else{
                printf("Something went wrong with video blur using shared memory\n");
                return -1;
            }
            break;
        }
        
    case Mode::SHARED_STREAMS:
        {
            int totalTimeWithStreams = -1;
            blurredVideo = GB.blurVideoGPUusingSharedMemoryAndStreams(video,&totalTimeWithStreams);
            if(totalTimeWithStreams!=-1){
                printf("GPU total time using streams and shared memory: %d milliseconds\n",totalTimeWithStreams);
            }
            else{
                printf("Something went wrong with video blur using shared memory and streams\n");
                return -1;
            }
            break;
        }
    }
    printf("-------------------\n");

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

    blurredVideo = Video(0,0,0,0,std::vector<unsigned char *>());

    return 0;

}