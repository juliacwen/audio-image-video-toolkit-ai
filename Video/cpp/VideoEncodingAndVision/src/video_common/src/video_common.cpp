/******************************************************************************
 * File: video_common.cpp
 * Description: Implementation of shared video primitives for motion estimation,
 *              frame prediction, residual computation, and stereo disparity.
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-05
 ******************************************************************************/

#include "../inc/video_common.h"
#include <chrono>
#include <ctime>
#include <cmath>
#include <iostream>

//=================== Global Variables ===================
LogLevel logLevel = LogLevel::NONE;
volatile std::sig_atomic_t stopFlag = 0;

//=================== Timestamp ===================
std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) % 1000;
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&in_time_t));
    std::string ts(buffer);
    ts += "." + std::to_string(ms.count());
    return ts;
}

//=================== Dense Optical Flow ===================
// Compute motion field between two grayscale frames using Farneback
cv::Mat computeVideoMotionField(const cv::Mat& prevGray, const cv::Mat& currGray) {
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(prevGray, currGray, flow,
                                 0.5, 3, 15, 3, 5, 1.2, 0);
    return flow;
}

//=================== Motion-compensated Frame Prediction ===================
cv::Mat predictNextVideoFrame(const cv::Mat& prevGray, const cv::Mat& flow) {
    cv::Mat predicted(prevGray.size(), prevGray.type(), cv::Scalar(0));
    for (int y = 0; y < prevGray.rows; y++) {
        for (int x = 0; x < prevGray.cols; x++) {
            cv::Point2f mv = flow.at<cv::Point2f>(y, x);
            int newX = cv::saturate_cast<int>(x + mv.x);
            int newY = cv::saturate_cast<int>(y + mv.y);
            if (0 <= newX && newX < prevGray.cols &&
                0 <= newY && newY < prevGray.rows) {
                predicted.at<uchar>(y, x) = prevGray.at<uchar>(newY, newX);
            }
        }
    }
    return predicted;
}

//=================== Compute Residual ===================
cv::Mat computeVideoResidual(const cv::Mat& currGray, const cv::Mat& predicted) {
    cv::Mat residual;
    cv::absdiff(currGray, predicted, residual);
    return residual;
}

//=================== Stereo Disparity ===================
cv::Mat computeVideoDisparity(const cv::Mat& leftGray, const cv::Mat& rightGray) {
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16, 9);
    cv::Mat disparity;
    stereo->compute(leftGray, rightGray, disparity);
    cv::normalize(disparity, disparity, 0, MAX_PIXEL_VALUE, cv::NORM_MINMAX, CV_8U);
    return disparity;
}

