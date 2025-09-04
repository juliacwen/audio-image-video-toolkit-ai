/******************************************************************************
 * File: video_motion_estimation.cpp
 * Description: Capture from camera, compute dense optical flow (Farneback),
 *              print representative motion statistics and write frames
 *              with motion-vector overlay to an MJPG/AVI file.
 *              Adds -v option for logging control with timestamp.
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-04
 ******************************************************************************/

#include <opencv2/opencv.hpp>
#include "../video_common/inc/video_common.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>

// Logging levels
enum LogLevel { NONE=0, INFO=1, WARNING=2, ERROR=3 };
LogLevel logLevel = NONE;

// Named constants
const int KEY_EXIT_CTRL_C = 3;
const int KEY_EXIT_ESC    = 27;

const float VECTOR_SCALE        = 3.0f;
const float HIGHLIGHT_THRESHOLD = 2.0f;
const int DISPLAY_STEP          = 15;
const int REF_POINT_RADIUS      = 1;

const float WINDOW_SCALE        = 0.5f; // second window scaling
const cv::Scalar STRONG_COLOR   = cv::Scalar(0,255,0);
const cv::Scalar NORMAL_COLOR   = cv::Scalar(0,255,255);
const cv::Scalar REF_COLOR      = cv::Scalar(0,0,255);

const std::vector<cv::Point> DEBUG_POINTS = { {50,50}, {100,100}, {200,150} };

// Timestamp helper
std::string currentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) % 1000;
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%H:%M:%S")
       << "." << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// Logging macro with timestamp
#define LOG(level, msg) \
    if (logLevel != NONE && level >= logLevel) { \
        std::cout << "[" << currentTimestamp() << "] " << msg << std::endl; \
    }

int main(int argc, char** argv) {
    // Parse -v argument
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-v" && i + 1 < argc) {
            int lvl = std::stoi(argv[i+1]);
            if (lvl >= 1 && lvl <= 3) logLevel = static_cast<LogLevel>(lvl);
        }
    }

    // Startup log
    LOG(INFO, "=== Video Motion Estimation ===");
    LOG(INFO, "Logging level: " << logLevel);

    // Open camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        LOG(ERROR, "Cannot open camera.");
        return -1;
    }
    LOG(INFO, "Camera opened.");

    cv::Mat prevFrame, prevGray;
    cap >> prevFrame;
    if (prevFrame.empty()) {
        LOG(ERROR, "Empty frame captured.");
        return -1;
    }
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

    // Video writer
    std::string outFile = "./output_video.avi";
    cv::VideoWriter writer(outFile,
                           cv::VideoWriter::fourcc('M','J','P','G'),
                           20,
                           prevFrame.size());
    if (!writer.isOpened()) {
        LOG(ERROR, "Could not open the video file for writing.");
        return -1;
    }
    LOG(INFO, "Writing to: " << outFile);

    cv::Mat currFrame, currGray;
    int frameCount = 0;

    while (true) {
        cap >> currFrame;
        if (currFrame.empty()) {
            LOG(WARNING, "Empty frame captured at frame " << frameCount << ". Skipping.");
            continue;
        }

        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

        cv::Mat flow = computeVideoMotionField(prevGray, currGray);

        cv::Mat display = currFrame.clone();
        cv::Mat displayScaled;
        cv::resize(display, displayScaled, cv::Size(), WINDOW_SCALE, WINDOW_SCALE);

        for (int y = 0; y < display.rows; y += DISPLAY_STEP) {
            for (int x = 0; x < display.cols; x += DISPLAY_STEP) {
                const cv::Point2f& fxy = flow.at<cv::Point2f>(y,x);
                float magnitude = std::sqrt(fxy.x*fxy.x + fxy.y*fxy.y);
                cv::Scalar color = (magnitude > HIGHLIGHT_THRESHOLD) ? STRONG_COLOR : NORMAL_COLOR;
                cv::line(display, cv::Point(x,y),
                         cv::Point(cvRound(x + fxy.x * VECTOR_SCALE), cvRound(y + fxy.y * VECTOR_SCALE)),
                         color, 2);
                cv::circle(display, cv::Point(x,y), REF_POINT_RADIUS, REF_COLOR, -1);
            }
        }

        // Debug log for points
        for (auto &pt : DEBUG_POINTS) {
            const cv::Point2f &f = flow.at<cv::Point2f>(pt.y, pt.x);
            LOG(INFO, "Frame " << frameCount << " motion(" << pt.x << "," << pt.y << "): dx=" << f.x << ", dy=" << f.y);
        }

        cv::imshow("Motion Vectors Full", display);
        cv::imshow("Motion Vectors Scaled", displayScaled);
        writer.write(display);

        int key = cv::waitKey(1);
        if (key == KEY_EXIT_CTRL_C || key == KEY_EXIT_ESC) break;

        prevGray = currGray.clone();
        frameCount++;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    LOG(INFO, "Video capture and writing completed: " << outFile);
    return 0;
}

