/******************************************************************************
 * File: video_common.cpp
 * Description: Implementation of shared video primitives for motion estimation,
 *              frame prediction, residual computation, and stereo disparity.
 * Author: [Your Name]
 * Date: 2025-08-29
 ******************************************************************************/

#include "video_common.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <ctime>

std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now_time));
    return std::string(buffer);
}


// Dense optical flow (Farneback)
cv::Mat computeVideoMotionField(const cv::Mat& prevGray, const cv::Mat& currGray) {
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(prevGray, currGray, flow,
                                 0.5, 3, 15, 3, 5, 1.2, 0);
    return flow;
}

// Motion-compensated frame prediction
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

// Compute residual
cv::Mat computeVideoResidual(const cv::Mat& currGray, const cv::Mat& predicted) {
    cv::Mat residual;
    cv::absdiff(currGray, predicted, residual);
    return residual;
}

// Stereo disparity (block matching)
cv::Mat computeVideoDisparity(const cv::Mat& leftGray, const cv::Mat& rightGray) {
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16, 9);
    cv::Mat disparity;
    stereo->compute(leftGray, rightGray, disparity);
    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8U);
    return disparity;
}

