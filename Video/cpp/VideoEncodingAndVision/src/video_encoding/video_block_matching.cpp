/******************************************************************************
 * File: video_block_matching.cpp
 * Description:
 *   Stereo disparity demonstration using classical block matching (StereoBM).
 *   NOTE: Uses ONLY ONE CAMERA. The "right" image is simulated by shifting
 *   the captured frame horizontally. The disparity results reflect only this
 *   artificial shift, NOT real-world depth.
 *
 *   Workflow:
 *     - Capture frame from camera
 *     - Convert to grayscale (left)
 *     - Create simulated "right" frame by shifting left
 *     - Run block matching (StereoBM)
 *     - Display original and disparity map in separate windows
 *     - Save disparity output with overlay (timestamp, FPS, frame count)
 *
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-04
 ******************************************************************************/

#include "../video_common/inc/video_common.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <ctime>
#include <csignal>

// Logging levels
enum LogLevel { NONE = 0, INFO = 1, WARNING = 2, ERROR = 3 };
LogLevel logLevel = NONE;

#define LOG(level, msg) \
    if (logLevel != NONE && level >= logLevel) { \
        std::cout << msg << std::endl; \
    }

// Constants to replace magic numbers
const int ESC_KEY = 27;
const int SHIFT_PIXELS = 4;
const int NUM_DISPARITIES = 16 * 5; // multiple of 16
const int BLOCK_SIZE = 15;
const int VIDEO_FPS = 20;
const int LOG_INTERVAL = 30;
const int MAX_PIXEL_VALUE = 255;

// Global flag for Ctrl-C
volatile std::sig_atomic_t stopFlag = 0;

// Signal handler for Ctrl-C
void handleSigInt(int) {
    stopFlag = 1;
}

int main(int argc, char** argv) {
    // Parse -v argument
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-v" && i + 1 < argc) {
            int lvl = std::stoi(argv[i+1]);
            if (lvl >= 1 && lvl <= 3) logLevel = static_cast<LogLevel>(lvl);
        }
    }

    // Register Ctrl-C signal
    std::signal(SIGINT, handleSigInt);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        LOG(ERROR, "[ERROR] Cannot open camera");
        return -1;
    }
    LOG(INFO, "[INFO] Camera opened successfully");

    cv::Mat frame, gray, grayShifted;

    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(NUM_DISPARITIES, BLOCK_SIZE);

    LOG(INFO, "=== BLOCK MATCHING INITIALIZED (StereoBM) ===");
    LOG(INFO, "[INFO] Block size = " << BLOCK_SIZE
              << ", search range = " << NUM_DISPARITIES << " disparities.");
    LOG(INFO, "[INFO] NOTE: Right image is simulated by shifting left image.");
    LOG(INFO, "=============================================");

    // Create windows explicitly
    cv::namedWindow("Original (Left)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Disparity Map", cv::WINDOW_AUTOSIZE);

    // VideoWriter
    cv::VideoWriter writer;
    bool writerInitialized = false;
    std::string outFile = "video_block_output.avi";

    int frameCount = 0;
    auto startTime = std::chrono::steady_clock::now();

    while (!stopFlag) {
        cap >> frame;
        if (frame.empty()) {
            LOG(WARNING, "[WARNING] Empty frame captured");
            continue;
        }

        frameCount++;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Simulated right frame
        cv::Mat translationMatrix = (cv::Mat_<double>(2,3) << 1, 0, SHIFT_PIXELS, 0, 1, 0);
        cv::warpAffine(gray, grayShifted, translationMatrix, gray.size());

        // Compute disparity
        cv::Mat disparity16S, disparity8U;
        stereoBM->compute(gray, grayShifted, disparity16S);
        disparity16S.convertTo(disparity8U, CV_8U, static_cast<double>(MAX_PIXEL_VALUE) / (NUM_DISPARITIES * 16.0));

        // Compute FPS
        auto elapsed = std::chrono::steady_clock::now() - startTime;
        double fps = frameCount / std::chrono::duration<double>(elapsed).count();

        // Overlay: timestamp, frame count, FPS
        std::string overlayText = getTimestamp() + " | Frame: " + std::to_string(frameCount) +
                                  " | FPS: " + cv::format("%.2f", fps);
        cv::putText(disparity8U, overlayText,
                    cv::Point(10, disparity8U.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(MAX_PIXEL_VALUE),
                    1, cv::LINE_AA);

        // Initialize writer
        if (!writerInitialized) {
            writer.open(outFile, cv::VideoWriter::fourcc('M','J','P','G'),
                        VIDEO_FPS, disparity8U.size(), false);
            if (!writer.isOpened()) {
                LOG(ERROR, "[ERROR] Could not open " << outFile << " for writing");
                return -1;
            }
            writerInitialized = true;
            LOG(INFO, "[INFO] VideoWriter initialized (" << outFile << ")");
        }

        // Write video
        writer.write(disparity8U);

        // Logging stats every LOG_INTERVAL frames
        if (frameCount % LOG_INTERVAL == 0) {
            double minVal, maxVal;
            cv::minMaxLoc(disparity16S, &minVal, &maxVal);
            LOG(INFO, "[FRAME " << frameCount << "] "
                      << "Resolution=" << frame.cols << "x" << frame.rows
                      << " | FPS=" << fps
                      << " | Disparity[min=" << minVal
                      << ", max=" << maxVal << "]");
        }

        // Display windows
        cv::imshow("Original (Left)", gray);
        cv::imshow("Disparity Map", disparity8U);

        char key = (char)cv::waitKey(1);
        if (key == ESC_KEY) {
            LOG(INFO, "[INFO] ESC pressed. Exiting...");
            break;
        }
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    LOG(INFO, "[INFO] Finished. Video saved to " << outFile);
    return 0;
}

