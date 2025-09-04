/******************************************************************************
 * File: video_residual.cpp
 * Description: Compute residual between current and predicted frames.
 *              - Motion estimation
 *              - Timestamp overlay
 *              - Two windows: Predicted + Residual
 *              - Logging with -v
 *              - Exit on ESC or Ctrl-C
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-04
 ******************************************************************************/

#include <opencv2/opencv.hpp>
#include "../video_common/inc/video_common.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <csignal>

// Logging levels
enum LogLevel { NONE = 0, INFO = 1, WARNING = 2, ERROR = 3 };
LogLevel logLevel = NONE;

#define LOG(level, msg) \
    if (logLevel != NONE && level >= logLevel) { std::cout << msg << std::endl; }

// Exit flag for Ctrl-C
volatile std::sig_atomic_t exitFlag = 0;
void signalHandler(int signal) { exitFlag = 1; }

// Constants
const int STEP = 15;
const int CIRCLE_RADIUS = 1;
const cv::Scalar ARROW_COLOR(0, 255, 0);
const cv::Scalar CIRCLE_COLOR(0, 255, 255);
const int ARROW_THICKNESS = 1;
const double FONT_SCALE = 0.5;
const int FONT_THICKNESS = 1;
const int ESC_KEY = 27;
const int FPS = 20;


int main(int argc, char** argv) {
    // Setup Ctrl-C handler
    std::signal(SIGINT, signalHandler);

    // Parse -v argument
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-v" && i + 1 < argc) {
            int lvl = std::stoi(argv[i + 1]);
            if (lvl >= 1 && lvl <= 3) logLevel = (LogLevel)lvl;
        }
    }

    std::cout << "=== Video Residual Computation ===" << std::endl;
    std::cout << "Logging level set to: " << logLevel << std::endl;
    std::cout << "Active logs: ";
    if (logLevel <= INFO) std::cout << "[INFO] ";
    if (logLevel <= WARNING) std::cout << "[WARNING] ";
    if (logLevel <= ERROR) std::cout << "[ERROR] ";
    std::cout << std::endl << "=================================" << std::endl;

    // Open default camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        LOG(ERROR, "[ERROR] Cannot open camera.");
        return -1;
    }
    LOG(INFO, "[INFO] Camera opened.");

    cv::Mat prevFrame, prevGray;
    cap >> prevFrame;
    if (prevFrame.empty()) {
        LOG(ERROR, "[ERROR] Empty frame captured at start.");
        return -1;
    }
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

    // Video writers
    std::string outPredFile = "./predicted_output.avi";
    std::string outResFile  = "./residual_output.avi";
    cv::VideoWriter writerPred(outPredFile, cv::VideoWriter::fourcc('M','J','P','G'), FPS, prevFrame.size(), true);
    cv::VideoWriter writerRes(outResFile, cv::VideoWriter::fourcc('M','J','P','G'), FPS, prevFrame.size(), false);

    if (!writerPred.isOpened() || !writerRes.isOpened()) {
        LOG(ERROR, "[ERROR] Could not open video writers.");
        return -1;
    }
    LOG(INFO, "[INFO] Video writers initialized.");

    int frameIdx = 0;
    cv::Mat currFrame, currGray;

    while (!exitFlag) {
        cap >> currFrame;
        if (currFrame.empty()) {
            LOG(WARNING, "[WARNING] Empty frame at index " << frameIdx);
            continue;
        }

        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

        // Motion field
        cv::Mat flow = computeVideoMotionField(prevGray, currGray);

        // Predict frame
        cv::Mat predicted = prevGray.clone();
        for (int y = 0; y < prevGray.rows; y++) {
            for (int x = 0; x < prevGray.cols; x++) {
                cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
                int newX = std::clamp(int(x + fxy.x), 0, prevGray.cols - 1);
                int newY = std::clamp(int(y + fxy.y), 0, prevGray.rows - 1);
                predicted.at<uchar>(y, x) = prevGray.at<uchar>(newY, newX);
            }
        }

        // Compute residual
        cv::Mat residual;
        cv::absdiff(predicted, currGray, residual);

        // Overlay motion vectors
        cv::Mat displayPred;
        cv::cvtColor(predicted, displayPred, cv::COLOR_GRAY2BGR);
        float maxDisp = 0.0f;
        for (int y = 0; y < flow.rows; y += STEP) {
            for (int x = 0; x < flow.cols; x += STEP) {
                cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
                float mag = std::sqrt(fxy.x*fxy.x + fxy.y*fxy.y);
                if (mag > maxDisp) maxDisp = mag;

                cv::arrowedLine(displayPred, cv::Point(x, y),
                                cv::Point(cv::saturate_cast<int>(x + fxy.x),
                                          cv::saturate_cast<int>(y + fxy.y)),
                                ARROW_COLOR, ARROW_THICKNESS, cv::LINE_AA);
                cv::circle(displayPred, cv::Point(x, y), CIRCLE_RADIUS, CIRCLE_COLOR, -1);

                LOG(INFO, "[INFO] Frame " << frameIdx << " motion(" << y << "," << x
                          << "): dx=" << fxy.x << ", dy=" << fxy.y);
            }
        }

        // Overlay timestamp and frame info
        std::string overlayText = getTimestamp() +
                                  " | Frame: " + std::to_string(frameIdx) +
                                  " | MaxDisp: " + cv::format("%.2f", maxDisp);
        cv::putText(displayPred, overlayText, cv::Point(10, displayPred.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, cv::Scalar(255,255,255), FONT_THICKNESS);

        // Show windows
        cv::imshow("Predicted Frame (overlay)", displayPred);
        cv::imshow("Residual Frame", residual);

        // Write videos
        writerPred.write(displayPred);
        writerRes.write(residual);

        prevGray = currGray.clone();
        frameIdx++;

        int key = cv::waitKey(1);
        if (key == ESC_KEY || exitFlag) break;
    }

    cap.release();
    writerPred.release();
    writerRes.release();
    cv::destroyAllWindows();

    LOG(INFO, "[INFO] Predicted frames written to " << outPredFile);
    LOG(INFO, "[INFO] Residual frames written to " << outResFile);

    return 0;
}

