/******************************************************************************
 * File: video_frame_prediction.cpp
 * Description:
 *   Demo program to perform motion-compensated frame prediction using dense
 *   optical flow. Adds -v option for timestamped logging control.
 *
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

// Key constants
const int KEY_EXIT_CTRL_C = 3;
const int KEY_EXIT_ESC    = 27;

// Motion visualization constants
const int SAMPLE_STEP      = 15;
const float WARNING_MAG    = 50.0f;
const int POINT_RADIUS     = 1;
const float WINDOW_SCALE   = 0.5f; // optional second window scaling

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

    // Open camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        LOG(ERROR, "Cannot open camera.");
        return -1;
    }
    LOG(INFO, "Camera opened successfully.");

    cv::Mat prevFrame, prevGray;
    cap >> prevFrame;
    if (prevFrame.empty()) {
        LOG(ERROR, "Empty frame captured at start.");
        return -1;
    }
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

    // Video writer
    std::string outFile = "./predicted_output.avi";
    cv::VideoWriter writer(outFile,
                           cv::VideoWriter::fourcc('M','J','P','G'),
                           20,
                           prevFrame.size());
    if (!writer.isOpened()) {
        LOG(ERROR, "Could not open the video file for writing.");
        return -1;
    }
    LOG(INFO, "VideoWriter initialized: " << outFile);

    cv::Mat currFrame, currGray;
    int frameIdx = 0;

    while (true) {
        cap >> currFrame;
        if (currFrame.empty()) {
            LOG(WARNING, "Empty frame captured at frame " << frameIdx << ". Skipping.");
            continue;
        }

        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

        // Compute motion field
        cv::Mat flow = computeVideoMotionField(prevGray, currGray);

        // Motion-compensated predicted frame
        cv::Mat predicted = prevGray.clone();
        for (int y = 0; y < prevGray.rows; y++) {
            for (int x = 0; x < prevGray.cols; x++) {
                cv::Point2f fxy = flow.at<cv::Point2f>(y,x);
                int newX = std::clamp(int(x + fxy.x), 0, prevGray.cols - 1);
                int newY = std::clamp(int(y + fxy.y), 0, prevGray.rows - 1);
                predicted.at<uchar>(y,x) = prevGray.at<uchar>(newY,newX);
            }
        }

        // Compute MAD and max displacement
        cv::Mat diff;
        cv::absdiff(predicted, currGray, diff);
        double mad = cv::mean(diff)[0];

        float maxDisp = 0.0f;
        for (int y = 0; y < flow.rows; y += SAMPLE_STEP) {
            for (int x = 0; x < flow.cols; x += SAMPLE_STEP) {
                cv::Point2f fxy = flow.at<cv::Point2f>(y,x);
                float mag = std::sqrt(fxy.x*fxy.x + fxy.y*fxy.y);
                if (mag > maxDisp) maxDisp = mag;
                if (mag > WARNING_MAG) {
                    LOG(WARNING, "Large motion magnitude at frame " << frameIdx << " (" << y << "," << x << ") = " << mag);
                }
            }
        }

        // Convert predicted to BGR for overlay
        cv::Mat predictedBGR;
        cv::cvtColor(predicted, predictedBGR, cv::COLOR_GRAY2BGR);

        // Draw motion vectors
        for (int y = 0; y < flow.rows; y += SAMPLE_STEP) {
            for (int x = 0; x < flow.cols; x += SAMPLE_STEP) {
                cv::Point2f fxy = flow.at<cv::Point2f>(y,x);
                cv::circle(predictedBGR, cv::Point(x,y), POINT_RADIUS, cv::Scalar(0,0,255), -1);
                cv::arrowedLine(predictedBGR,
                                cv::Point(x,y),
                                cv::Point(x + fxy.x, y + fxy.y),
                                cv::Scalar(0,255,0), 1, cv::LINE_AA);
            }
        }

        std::string overlayText = "Frame: " + std::to_string(frameIdx) +
                                  " | MAD: " + cv::format("%.2f", mad) +
                                  " | MaxDisp: " + cv::format("%.2f", maxDisp);
        cv::putText(predictedBGR, overlayText,
                    cv::Point(10, predictedBGR.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255,255,255), 1, cv::LINE_AA);

        // Show two windows
        cv::imshow("Predicted Frame", predictedBGR);
        cv::Mat predictedSmall;
        cv::resize(predictedBGR, predictedSmall, cv::Size(), WINDOW_SCALE, WINDOW_SCALE);
        cv::imshow("Predicted Frame Scaled", predictedSmall);

        writer.write(predictedBGR);
        prevGray = currGray.clone();
        frameIdx++;

        int key = cv::waitKey(1);
        if (key == KEY_EXIT_CTRL_C || key == KEY_EXIT_ESC) break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    LOG(INFO, "Predicted frames with motion vectors written to " << outFile);
    return 0;
}

