/******************************************************************************
 * File: video_block_matching.cpp
 * Description:
 *   Stereo disparity demo using classical block matching (StereoBM).
 *   Right image is simulated by shifting the left image horizontally.
 *   Disparity results reflect this artificial shift, NOT real depth.
 * Author: Julia Wen
 * Date: 2025-09-05
 ******************************************************************************/

#include "../video_common/inc/video_common.h"
#include <opencv2/opencv.hpp>
#include <iostream>

inline constexpr int SHIFT_PIXELS = 4;            // artificial stereo baseline

int main(int argc, char** argv) {
    // Parse -v argument for logging
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-v" && i + 1 < argc) {
            int lvl = std::stoi(argv[i + 1]);
            if (lvl >= static_cast<int>(LogLevel::INFO) &&
                lvl <= static_cast<int>(LogLevel::ERROR))
            {
                logLevel = static_cast<LogLevel>(lvl);
            }
        }
    }

    // Register Ctrl-C
    std::signal(SIGINT, handleSigInt);

    LOG(LogLevel::INFO, "=== BLOCK MATCHING INITIALIZED (StereoBM) ===");

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        LOG(LogLevel::ERROR, "Cannot open camera");
        return -1;
    }
    LOG(LogLevel::INFO, "Camera opened successfully");

    cv::Mat frame, gray, grayShifted;

    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(NUM_DISPARITIES, BLOCK_SIZE);
    LOG(LogLevel::INFO, "Block size = " << BLOCK_SIZE
              << ", search range = " << NUM_DISPARITIES
              << " disparities. NOTE: Right image simulated by shifting left image.");

    cv::namedWindow("Original (Left)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Disparity Map", cv::WINDOW_AUTOSIZE);

    cv::VideoWriter writer;
    bool writerInitialized = false;
    std::string outFile = "video_block_output.avi";

    int frameCount = 0;
    auto startTime = std::chrono::steady_clock::now();

    while (!stopFlag) {
        cap >> frame;
        if (frame.empty()) {
            LOG(LogLevel::WARNING, "Empty frame captured");
            continue;
        }

        ++frameCount;
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
                LOG(LogLevel::ERROR, "Could not open " << outFile << " for writing");
                return -1;
            }
            writerInitialized = true;
            LOG(LogLevel::INFO, "VideoWriter initialized (" << outFile << ")");
        }

        // Write video
        writer.write(disparity8U);

        // Logging stats every LOG_INTERVAL frames
        if (frameCount % LOG_INTERVAL == 0) {
            double minVal, maxVal;
            cv::minMaxLoc(disparity16S, &minVal, &maxVal);
            LOG(LogLevel::INFO, "[FRAME " << frameCount << "] "
                      << "Resolution=" << frame.cols << "x" << frame.rows
                      << " | FPS=" << fps
                      << " | Disparity[min=" << minVal
                      << ", max=" << maxVal << "]");
        }

        // Display windows
        cv::imshow("Original (Left)", gray);
        cv::imshow("Disparity Map", disparity8U);

        char key = (char)cv::waitKey(1);
        if (key == KEY_EXIT_ESC) {
            LOG(LogLevel::INFO, "ESC pressed. Exiting...");
            break;
        }
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    LOG(LogLevel::INFO, "Finished. Video saved to " << outFile);

    return 0;
}

