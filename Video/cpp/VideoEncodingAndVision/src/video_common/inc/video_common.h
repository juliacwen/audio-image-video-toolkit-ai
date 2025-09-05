/******************************************************************************
 * File: video_common.h
 * Description: Shared video primitives for motion estimation, frame prediction,
 *              residual computation, and stereo disparity. Usable by both
 *              encoding demos and motion/depth applications.
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-05
 ******************************************************************************/

#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <csignal>
#include <atomic>

//=================== Constants ===================
inline constexpr int DISPLAY_STEP      = 15;
inline constexpr int SAMPLE_STEP       = 15;    // Step for motion vector sampling
inline constexpr float VECTOR_SCALE    = 3.0f;  // Scaling factor for motion visualization
inline constexpr int REF_POINT_RADIUS  = 1;     // Reference point radius for overlay
inline constexpr int MAX_PIXEL_VALUE   = 255;   // Maximum value for 8-bit images

inline constexpr int KEY_EXIT_CTRL_C   = 3;
inline constexpr int KEY_EXIT_ESC      = 27;

inline constexpr float HIGHLIGHT_THRESHOLD = 2.0f;        // Threshold for strong motion highlight
inline constexpr float WINDOW_SCALE        = 0.5f;        // Scale factor for optional second window
inline const cv::Scalar STRONG_COLOR      = cv::Scalar(0,255,0);
inline const cv::Scalar NORMAL_COLOR      = cv::Scalar(0,255,255);
inline const cv::Scalar REF_COLOR         = cv::Scalar(0,0,255);

inline constexpr int NUM_DISPARITIES = 16 * 5;    // must be multiple of 16
inline constexpr int BLOCK_SIZE = 15;             // block matching window size
inline constexpr int VIDEO_FPS = 20;              // output video FPS
inline constexpr int LOG_INTERVAL = 30;           // log stats every N frames


inline const std::vector<cv::Point> DEBUG_POINTS = { {50,50}, {100,100}, {200,150} };

//=================== Logging ===================
enum class LogLevel { NONE=0, INFO=1, WARNING=2, ERROR=3 };
extern LogLevel logLevel;

#define LOG(level, msg) \
    if (logLevel != LogLevel::NONE && static_cast<int>(level) >= static_cast<int>(logLevel)) { \
        std::cout << "[" << getTimestamp() << "] " << msg << std::endl; \
    }

//=================== Signal Handling ===================
extern volatile std::sig_atomic_t stopFlag;
inline void handleSigInt(int) { stopFlag = 1; }

//=================== Function Declarations ===================

// Return current timestamp as string
std::string getTimestamp();

// Compute dense optical flow (motion field) between two grayscale frames
cv::Mat computeVideoMotionField(const cv::Mat& prevGray, const cv::Mat& currGray);

// Predict next frame using motion-compensated flow
cv::Mat predictNextVideoFrame(const cv::Mat& prevGray, const cv::Mat& flow);

// Compute residual between current frame and predicted frame
cv::Mat computeVideoResidual(const cv::Mat& currGray, const cv::Mat& predicted);

// Compute disparity map between left and right grayscale images
cv::Mat computeVideoDisparity(const cv::Mat& leftGray, const cv::Mat& rightGray);

