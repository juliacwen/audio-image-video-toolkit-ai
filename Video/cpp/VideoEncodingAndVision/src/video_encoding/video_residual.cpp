/******************************************************************************
 * File: video_residual.cpp
 * Description: Compute residual between current and predicted frames.
 *              - Motion estimation
 *              - Timestamp overlay
 *              - Two windows: Predicted + Residual
 *              - Logging with -v
 *              - Exit on ESC or Ctrl-C
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-05
 ******************************************************************************/

#include <opencv2/opencv.hpp>
#include "../video_common/inc/video_common.h"
#include <iostream>
#include <cmath>
#include <csignal>

int main(int argc, char** argv) {
    // Local visual constants (kept local to avoid polluting the common header)
    constexpr int kArrowThickness = 1;
    constexpr double kFontScale = 0.5;
    constexpr int kFontThickness = 1;

    // Register Ctrl-C handler defined in video_common.h
    std::signal(SIGINT, handleSigInt);

    // Parse -v argument (use LogLevel from video_common.h)
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-v" && i + 1 < argc) {
            int lvl = std::stoi(argv[i + 1]);
            if (lvl >= static_cast<int>(LogLevel::INFO) && lvl <= static_cast<int>(LogLevel::ERROR)) {
                logLevel = static_cast<LogLevel>(lvl);
            }
        }
    }

    LOG(LogLevel::INFO, "=== Video Residual Computation ===");
    LOG(LogLevel::INFO, "Logging level: " << static_cast<int>(logLevel));

    // Open default camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        LOG(LogLevel::ERROR, "Cannot open camera.");
        return -1;
    }
    LOG(LogLevel::INFO, "Camera opened.");

    cv::Mat prevFrame, prevGray;
    cap >> prevFrame;
    if (prevFrame.empty()) {
        LOG(LogLevel::ERROR, "Empty frame captured at start.");
        return -1;
    }
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

    // Video writers: predicted (color) and residual (grayscale)
    const std::string outPredFile = "./predicted_output.avi";
    const std::string outResFile  = "./residual_output.avi";
    cv::VideoWriter writerPred(outPredFile,
                               cv::VideoWriter::fourcc('M','J','P','G'),
                               VIDEO_FPS,
                               prevFrame.size(), /*isColor=*/true);
    cv::VideoWriter writerRes(outResFile,
                              cv::VideoWriter::fourcc('M','J','P','G'),
                              VIDEO_FPS,
                              prevFrame.size(), /*isColor=*/false);

    if (!writerPred.isOpened() || !writerRes.isOpened()) {
        LOG(LogLevel::ERROR, "Could not open video writers.");
        return -1;
    }
    LOG(LogLevel::INFO, "Video writers initialized.");

    int frameIdx = 0;
    cv::Mat currFrame, currGray;

    while (!stopFlag) {
        cap >> currFrame;
        if (currFrame.empty()) {
            LOG(LogLevel::WARNING, "Empty frame at index " << frameIdx);
            continue;
        }

        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

        // Compute motion field
        cv::Mat flow = computeVideoMotionField(prevGray, currGray);

        // Predict frame using motion-compensated flow
        cv::Mat predicted = predictNextVideoFrame(prevGray, flow);

        // Compute residual between current and predicted
        cv::Mat residual = computeVideoResidual(currGray, predicted);

        // Overlay motion vectors on predicted frame
        cv::Mat displayPred;
        cv::cvtColor(predicted, displayPred, cv::COLOR_GRAY2BGR);
        float maxDisp = 0.0f;

        for (int y = 0; y < flow.rows; y += SAMPLE_STEP) {
            for (int x = 0; x < flow.cols; x += SAMPLE_STEP) {
                cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
                float mag = std::sqrt(fxy.x * fxy.x + fxy.y * fxy.y);
                if (mag > maxDisp) maxDisp = mag;

                // Draw arrow scaled by VECTOR_SCALE
                cv::Point p0(x, y);
                cv::Point p1(
                    cv::saturate_cast<int>(x + fxy.x * VECTOR_SCALE),
                    cv::saturate_cast<int>(y + fxy.y * VECTOR_SCALE)
                );
                cv::arrowedLine(displayPred, p0, p1, STRONG_COLOR, kArrowThickness, cv::LINE_AA);
                cv::circle(displayPred, p0, REF_POINT_RADIUS, REF_COLOR, -1);

                LOG(LogLevel::INFO, "Frame " << frameIdx << " motion(" << y << "," << x
                                             << "): dx=" << fxy.x << ", dy=" << fxy.y);
            }
        }

        // Overlay timestamp and frame info
        std::string overlayText = getTimestamp() +
                                  " | Frame: " + std::to_string(frameIdx) +
                                  " | MaxDisp: " + cv::format("%.2f", maxDisp);
        cv::putText(displayPred, overlayText,
                    cv::Point(10, displayPred.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    kFontScale,
                    cv::Scalar(MAX_PIXEL_VALUE, MAX_PIXEL_VALUE, MAX_PIXEL_VALUE),
                    kFontThickness);

        // Show windows
        cv::imshow("Predicted Frame (overlay)", displayPred);
        cv::imshow("Residual Frame", residual);

        // Write videos
        writerPred.write(displayPred);
        writerRes.write(residual);

        prevGray = currGray.clone();
        ++frameIdx;

        int key = cv::waitKey(1);
        if (key == KEY_EXIT_ESC || stopFlag) break;
    }

    cap.release();
    writerPred.release();
    writerRes.release();
    cv::destroyAllWindows();

    LOG(LogLevel::INFO, "Predicted frames written to " << outPredFile);
    LOG(LogLevel::INFO, "Residual frames written to " << outResFile);

    return 0;
}

