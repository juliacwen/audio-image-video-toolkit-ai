/******************************************************************************
 * File: video_common.cpp
 * Description: Implementation of shared video primitives for motion estimation,
 *              frame prediction, residual computation, and stereo disparity.
 * Author: Julia Wen (wendigilane@gmail.com)
 * 09-05-2025 — Initial check-in  
 * 11-30-2025 — improvement
 ******************************************************************************/

#include "../inc/video_common.h"
#include <chrono>
#include <iomanip>
#include <sstream>

namespace video_common {

// Define global variables
LogLevel logLevel = LogLevel::NONE;
std::atomic<bool> stopFlag{false};

//=================== Timestamp ===================
std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

//=================== Dense Optical Flow ===================
// Compute motion field between two grayscale frames using Farneback
cv::Mat computeVideoMotionField(const cv::Mat& prevGray, const cv::Mat& currGray) {
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(
        prevGray, currGray, flow,
        0.5,    // pyr_scale
        3,      // levels
        15,     // winsize
        3,      // iterations
        5,      // poly_n
        1.2,    // poly_sigma
        0       // flags
    );
    return flow;
}

//=================== Motion-compensated Frame Prediction ===================
cv::Mat predictNextVideoFrame(const cv::Mat& prevGray, const cv::Mat& flow) {
    cv::Mat predicted = cv::Mat::zeros(prevGray.size(), prevGray.type());
    
    for (int y = 0; y < prevGray.rows; ++y) {
        for (int x = 0; x < prevGray.cols; ++x) {
            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
            int newX = cvRound(x + fxy.x);
            int newY = cvRound(y + fxy.y);
            
            if (newX >= 0 && newX < prevGray.cols && newY >= 0 && newY < prevGray.rows) {
                predicted.at<uchar>(newY, newX) = prevGray.at<uchar>(y, x);
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
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(NUM_DISPARITIES, BLOCK_SIZE);
    cv::Mat disparity;
    stereo->compute(leftGray, rightGray, disparity);
    
    // Normalize for display
    cv::normalize(disparity, disparity, 0, MAX_PIXEL_VALUE, cv::NORM_MINMAX, CV_8U);
    return disparity;
}

} // namespace video_common
