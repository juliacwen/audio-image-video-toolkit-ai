/******************************************************************************
 * File: video_common.h
 * Description: Shared video primitives for motion estimation, frame prediction,
 *              residual computation, and stereo disparity. Usable by both
 *              encoding demos and motion/depth applications.
 * Author: [Your Name]
 * Date: 2025-08-29
 ******************************************************************************/

#pragma once
#include <opencv2/opencv.hpp>
#pragma once
#include <string>

std::string getTimestamp();

// Compute dense optical flow (motion field) between two grayscale frames
cv::Mat computeVideoMotionField(const cv::Mat& prevGray, const cv::Mat& currGray);

// Predict next frame using motion-compensated flow
cv::Mat predictNextVideoFrame(const cv::Mat& prevGray, const cv::Mat& flow);

// Compute residual between current frame and predicted frame
cv::Mat computeVideoResidual(const cv::Mat& currGray, const cv::Mat& predicted);

// Compute disparity map between left and right grayscale images
cv::Mat computeVideoDisparity(const cv::Mat& leftGray, const cv::Mat& rightGray);

