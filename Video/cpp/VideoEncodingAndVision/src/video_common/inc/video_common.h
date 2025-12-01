/******************************************************************************
 * File: video_common.h
 * Description: Shared video primitives for motion estimation, frame prediction,
 *              residual computation, and stereo disparity. Usable by both
 *              encoding demos and motion/depth applications.
 * Author: Julia Wen (wendigilane@gmail.com)
 * 09-05-2025 — Initial check-in  
 * 11-30-2025 — improvement
 ******************************************************************************/

#ifndef VIDEO_COMMON_H
#define VIDEO_COMMON_H
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <csignal>
#include <atomic>
#include <vector>
#include <iostream>

namespace video_common {

//=================== Constants ===================
constexpr int DISPLAY_STEP      = 15;
constexpr int SAMPLE_STEP       = 15;    // Step for motion vector sampling
constexpr float VECTOR_SCALE    = 3.0f;  // Scaling factor for motion visualization
constexpr int REF_POINT_RADIUS  = 1;     // Reference point radius for overlay
constexpr int MAX_PIXEL_VALUE   = 255;   // Maximum value for 8-bit images
constexpr int KEY_EXIT_CTRL_C   = 3;
constexpr int KEY_EXIT_ESC      = 27;
constexpr float HIGHLIGHT_THRESHOLD = 2.0f;        // Threshold for strong motion highlight
constexpr float HIGHLIGHT_THRESHOLD_SQ = HIGHLIGHT_THRESHOLD * HIGHLIGHT_THRESHOLD; // Squared threshold for performance
constexpr float WINDOW_SCALE        = 0.5f;        // Scale factor for optional second window

const cv::Scalar STRONG_COLOR      = cv::Scalar(0, 255, 0);
const cv::Scalar NORMAL_COLOR      = cv::Scalar(0, 255, 255);
const cv::Scalar REF_COLOR         = cv::Scalar(0, 0, 255);

constexpr int NUM_DISPARITIES = 16 * 5;    // must be multiple of 16
constexpr int BLOCK_SIZE = 15;             // block matching window size
constexpr int VIDEO_FPS = 20;              // output video FPS
constexpr int LOG_INTERVAL = 30;           // log stats every N frames
constexpr int MAX_EMPTY_FRAMES = 10;       // maximum consecutive empty frames before exit

const std::vector<cv::Point> DEBUG_POINTS = { {50, 50}, {100, 100}, {200, 150} };

//=================== Logging ===================
enum class LogLevel { NONE = 0, INFO = 1, WARNING = 2, ERROR = 3 };

extern LogLevel logLevel;

// Thread-safe logging macro with do-while wrapper
#define LOG(level, msg) \
    do { \
        if (video_common::logLevel != video_common::LogLevel::NONE && \
            static_cast<int>(level) >= static_cast<int>(video_common::logLevel)) { \
            std::cout << "[" << video_common::getTimestamp() << "] " << msg << std::endl; \
        } \
    } while(0)

//=================== Signal Handling ===================
extern std::atomic<bool> stopFlag;

inline void handleSigInt(int) { 
    stopFlag.store(true, std::memory_order_release); 
}

//=================== Function Declarations ===================

/**
 * @brief Get current timestamp as formatted string
 * @return Timestamp string in format "YYYY-MM-DD HH:MM:SS"
 */
std::string getTimestamp();

/**
 * @brief Compute dense optical flow (motion field) between two grayscale frames
 * @param prevGray Previous frame in grayscale
 * @param currGray Current frame in grayscale
 * @return Dense optical flow field (CV_32FC2)
 */
cv::Mat computeVideoMotionField(const cv::Mat& prevGray, const cv::Mat& currGray);

/**
 * @brief Predict next frame using motion-compensated flow
 * @param prevGray Previous grayscale frame
 * @param flow Optical flow field
 * @return Predicted frame
 */
cv::Mat predictNextVideoFrame(const cv::Mat& prevGray, const cv::Mat& flow);

/**
 * @brief Compute residual between current frame and predicted frame
 * @param currGray Current grayscale frame
 * @param predicted Predicted frame
 * @return Residual frame
 */
cv::Mat computeVideoResidual(const cv::Mat& currGray, const cv::Mat& predicted);

/**
 * @brief Compute disparity map between left and right grayscale images
 * @param leftGray Left camera grayscale image
 * @param rightGray Right camera grayscale image
 * @return Disparity map
 */
cv::Mat computeVideoDisparity(const cv::Mat& leftGray, const cv::Mat& rightGray);

/**
 * @brief Validate if a point is within image bounds
 * @param pt Point to validate
 * @param rows Image height
 * @param cols Image width
 * @return true if point is valid, false otherwise
 */
inline bool isPointValid(const cv::Point& pt, int rows, int cols) {
    return pt.x >= 0 && pt.x < cols && pt.y >= 0 && pt.y < rows;
}

} // namespace video_common

#endif // VIDEO_COMMON_H