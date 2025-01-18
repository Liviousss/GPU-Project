#include <opencv2/opencv.hpp>
#include <iostream>

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
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        // Allocate memory for the frame
        unsigned char* frameData = new unsigned char[frameSize];

        // Copy frame data into the allocated memory
        std::memcpy(frameData, frame.data, frameSize);

        // Store the pointer
        frames.push_back(frameData);
    }


    // Release resources
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
