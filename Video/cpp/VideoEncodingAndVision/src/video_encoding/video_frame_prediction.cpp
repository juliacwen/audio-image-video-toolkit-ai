/******************************************************************************
 * File: video_frame_prediction.cpp
 * Description:
 *   Demo program to perform motion-compensated frame prediction using dense
 *   optical flow. Uses constants and LOG from video_common.
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-05
 ******************************************************************************/

#include "../video_common/inc/video_common.h"
#include <opencv2/opencv.hpp>
#include <iostream>

// Main program
int main(int argc, char** argv) {
    // Parse -v argument for logging level
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-v" && i + 1 < argc) {
            int lvl = std::stoi(argv[i+1]);
            if (lvl >= static_cast<int>(LogLevel::INFO) && lvl <= static_cast<int>(LogLevel::ERROR)) {
                logLevel = static_cast<LogLevel>(lvl);
            }
        }
    }

    LOG(LogLevel::INFO, "=== Video Frame Prediction ===");

    // Open camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        LOG(LogLevel::ERROR, "Cannot open camera.");
        return -1;
    }
    LOG(LogLevel::INFO, "Camera opened successfully.");

    cv::Mat prevFrame, prevGray;
    cap >> prevFrame;
    if (prevFrame.empty()) {
        LOG(LogLevel::ERROR, "Empty frame captured at start.");
        return -1;
    }
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

    // Video writer
    std::string outFile = "./predicted_output.avi";
    cv::VideoWriter writer(outFile,
                           cv::VideoWriter::fourcc('M','J','P','G'),
                           VIDEO_FPS,
                           prevFrame.size());
    if (!writer.isOpened()) {
        LOG(LogLevel::ERROR, "Could not open video file for writing.");
        return -1;
    }
    LOG(LogLevel::INFO, "VideoWriter initialized: " << outFile);

    cv::Mat currFrame, currGray;
    int frameIdx = 0;

    while (!stopFlag) {
        cap >> currFrame;
        if (currFrame.empty()) {
            LOG(LogLevel::WARNING, "Empty frame at index " << frameIdx << ". Skipping.");
            continue;
        }

        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

        // Compute motion field
        cv::Mat flow = computeVideoMotionField(prevGray, currGray);

        // Motion-compensated predicted frame
        cv::Mat predicted = predictNextVideoFrame(prevGray, flow);

        // Compute mean absolute difference
        cv::Mat diff;
        cv::absdiff(predicted, currGray, diff);
        double mad = cv::mean(diff)[0];

        // Compute max displacement and log large motions
        float maxDisp = 0.0f;
        for (int y = 0; y < flow.rows; y += SAMPLE_STEP) {
            for (int x = 0; x < flow.cols; x += SAMPLE_STEP) {
                cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
                float mag = std::sqrt(fxy.x*fxy.x + fxy.y*fxy.y);
                if (mag > maxDisp) maxDisp = mag;
                if (mag > HIGHLIGHT_THRESHOLD) {
                    LOG(LogLevel::WARNING, "Large motion at frame " << frameIdx
                        << " (" << y << "," << x << ") = " << mag);
                }
            }
        }

        // Convert predicted to BGR for overlay
        cv::Mat predictedBGR;
        cv::cvtColor(predicted, predictedBGR, cv::COLOR_GRAY2BGR);

        // Draw motion vectors
        for (int y = 0; y < flow.rows; y += SAMPLE_STEP) {
            for (int x = 0; x < flow.cols; x += SAMPLE_STEP) {
                cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
                cv::circle(predictedBGR, cv::Point(x, y), REF_POINT_RADIUS, REF_COLOR, -1);
                cv::arrowedLine(predictedBGR,
                                cv::Point(x, y),
                                cv::Point(cvRound(x + fxy.x * VECTOR_SCALE), cvRound(y + fxy.y * VECTOR_SCALE)),
                                STRONG_COLOR, 1, cv::LINE_AA);
            }
        }

        // Overlay text
        std::string overlayText = "Frame: " + std::to_string(frameIdx) +
                                  " | MAD: " + cv::format("%.2f", mad) +
                                  " | MaxDisp: " + cv::format("%.2f", maxDisp);
        cv::putText(predictedBGR, overlayText,
                    cv::Point(10, predictedBGR.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(MAX_PIXEL_VALUE, MAX_PIXEL_VALUE, MAX_PIXEL_VALUE),
                    1, cv::LINE_AA);

        // Display windows
        cv::Mat predictedSmall;
        cv::resize(predictedBGR, predictedSmall, cv::Size(), WINDOW_SCALE, WINDOW_SCALE);
        cv::imshow("Predicted Frame", predictedBGR);
        cv::imshow("Predicted Frame Scaled", predictedSmall);

        // Write video
        writer.write(predictedBGR);

        prevGray = currGray.clone();
        frameIdx++;

        int key = cv::waitKey(1);
        if (key == KEY_EXIT_CTRL_C || key == KEY_EXIT_ESC) break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    LOG(LogLevel::INFO, "Predicted frames written to " << outFile);
    return 0;
}

