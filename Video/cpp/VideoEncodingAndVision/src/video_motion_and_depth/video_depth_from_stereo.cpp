/******************************************************************************
 * File: video_depth_from_stereo.cpp
 * Description: Fully-featured stereo depth demo with logging, grayscale + colorized disparity.
 *              Simulates right camera if only one camera is available (pixel shift).
 *              Logs per-frame metrics and run summary.
 * Return: 0  -> success
 *         -1 -> error (e.g., camera or log file could not be opened)
 * Author: [Your Name]
 * Date: 2025-09-04
 ******************************************************************************/

#include <opencv2/opencv.hpp>
#include "../video_common/inc/video_common.h"
#include <iostream>
#include <fstream>

// =================== Constants ===================
const int DEFAULT_SHIFT_PIXELS = 5;           // Default simulated right camera shift
const double DEFAULT_FALLBACK_FPS = 20.0;    // Fallback FPS if camera does not report
const int VIDEO_CODEC = cv::VideoWriter::fourcc('M','J','P','G'); // Video codec
const int ESC_KEY = 27;                       // ESC key to quit
const double DISPARITY_SCALE = 255.0;        // Scale for disparity normalization

// Output filenames
const std::string DEPTH_FILE = "depth_output.avi";          
const std::string DEPTH_COLOR_FILE = "depth_color_output.avi";
const std::string LEFT_FILE = "left_output.avi";
const std::string RIGHT_FILE = "right_output.avi";
// ==================================================

// Helper: normalize and colorize disparity
cv::Mat normalizeAndColorizeDisparity(const cv::Mat& disparity) {
    double minVal, maxVal;
    cv::minMaxLoc(disparity, &minVal, &maxVal);
    cv::Mat normalizedDisp, colorDisp;
    disparity.convertTo(normalizedDisp, CV_8U, DISPARITY_SCALE / (maxVal - minVal),
                        -minVal * DISPARITY_SCALE / (maxVal - minVal));
    cv::applyColorMap(normalizedDisp, colorDisp, cv::COLORMAP_JET);
    return colorDisp;
}

std::string getDateTime() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// =================== Main ===================
int main(int argc, char** argv) {
    bool verbose = false;
    int shift = DEFAULT_SHIFT_PIXELS;

    // Parse command-line args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-v") {
            verbose = true;
        } else if (arg == "-s" && i + 1 < argc) {
            shift = std::stoi(argv[++i]);
        } else {
            std::cout << "Usage: " << argv[0] << " [-v] [-s shift_pixels]" << std::endl;
            return 0;
        }
    }

    // Open left camera
    cv::VideoCapture capLeft(0);
    if (!capLeft.isOpened()) {
        std::cerr << "Error: Cannot open camera." << std::endl;
        return -1;
    }

    // Camera properties
    int camWidth  = static_cast<int>(capLeft.get(cv::CAP_PROP_FRAME_WIDTH));
    int camHeight = static_cast<int>(capLeft.get(cv::CAP_PROP_FRAME_HEIGHT));
    double camFps = capLeft.get(cv::CAP_PROP_FPS);
    if (camFps <= 0) camFps = DEFAULT_FALLBACK_FPS;

    // Log file
    std::ofstream logFile("video_depth_log.txt");
    if (!logFile.is_open()) {
        std::cerr << "Error: Cannot open log file." << std::endl;
        return -1;
    }

    auto runStart = std::chrono::steady_clock::now();

    // Header
    logFile << "===== Stereo Depth Run =====" << std::endl;
    logFile << "Start Time : " << getDateTime() << std::endl;
    logFile << "Shift      : " << shift << " px" << std::endl;
    logFile << "Verbose    : " << (verbose ? "ON" : "OFF") << std::endl;
    logFile << "Resolution : " << camWidth << "x" << camHeight << std::endl;
    logFile << "Camera FPS : " << camFps << std::endl;
    logFile << "Output     : " << DEPTH_FILE       << " (grayscale disparity)" << std::endl;
    logFile << "           : " << DEPTH_COLOR_FILE << " (colorized disparity)" << std::endl;
    logFile << "           : " << LEFT_FILE        << " (left camera)" << std::endl;
    logFile << "           : " << RIGHT_FILE       << " (simulated right)" << std::endl;
    logFile << "============================" << std::endl;

    // Mats and video writers
    cv::Mat leftFrame, rightFrame, leftGray, rightGray, disparity, disparityColor;
    cv::VideoWriter writerDisp, writerLeft, writerRight, writerDispColor;
    bool writerInitialized = false;

    long frameIndex = 0;
    double sumDisparity = 0.0;

    while (true) {
        capLeft >> leftFrame;
        if (leftFrame.empty()) break;
        frameIndex++;

        // Simulate right camera
        cv::Mat shiftMat = (cv::Mat_<double>(2,3) << 1, 0, shift, 0, 1, 0);
        cv::warpAffine(leftFrame, rightFrame, shiftMat, leftFrame.size());

        // Grayscale conversion
        cv::cvtColor(leftFrame, leftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightFrame, rightGray, cv::COLOR_BGR2GRAY);

        // Compute disparity
        disparity = computeVideoDisparity(leftGray, rightGray);

        // Colorize disparity using helper
        disparityColor = normalizeAndColorizeDisparity(disparity);

        // Initialize video writers once
        if (!writerInitialized) {
            double fps = DEFAULT_FALLBACK_FPS;
            writerDisp.open(DEPTH_FILE, VIDEO_CODEC, fps, disparity.size(), false);
            writerDispColor.open(DEPTH_COLOR_FILE, VIDEO_CODEC, fps, disparityColor.size(), true);
            writerLeft.open(LEFT_FILE, VIDEO_CODEC, fps, leftFrame.size(), true);
            writerRight.open(RIGHT_FILE, VIDEO_CODEC, fps, rightFrame.size(), true);

            if (!writerDisp.isOpened() || !writerDispColor.isOpened() ||
                !writerLeft.isOpened() || !writerRight.isOpened()) {
                std::cerr << "Error: Cannot open one of the video writers." << std::endl;
                return -1;
            }
            writerInitialized = true;
        }

        // Write videos
        writerDisp.write(disparity);
        writerDispColor.write(disparityColor);
        writerLeft.write(leftFrame);
        writerRight.write(rightFrame);

        // Log per-frame
        cv::Scalar meanDisp = cv::mean(disparity);
        sumDisparity += meanDisp[0];
        std::string ts = getTimestamp();
        logFile << "Frame " << frameIndex << " @ [" << ts << "] avg disparity: " << meanDisp[0] << std::endl;
        if (verbose) std::cout << "Frame " << frameIndex << " @ [" << ts << "] avg disparity: " << meanDisp[0] << std::endl;

        // Display
        cv::imshow("Depth / Disparity Map (Colorized)", disparityColor);
        if (cv::waitKey(1) == ESC_KEY) break;
    }

    // Release resources
    capLeft.release();
    writerDisp.release();
    writerDispColor.release();
    writerLeft.release();
    writerRight.release();

    // Footer summary
    auto runEnd = std::chrono::steady_clock::now();
    double durationSec = std::chrono::duration_cast<std::chrono::milliseconds>(runEnd - runStart).count() / 1000.0;
    double avgDisparity = (frameIndex > 0) ? sumDisparity / frameIndex : 0.0;

    logFile << "===== Run Summary =====" << std::endl;
    logFile << "Total Frames Captured : " << frameIndex << std::endl;
    logFile << "Run Duration          : " << durationSec << " seconds" << std::endl;
    logFile << "Average Disparity     : " << avgDisparity << std::endl;
    logFile << "=======================" << std::endl;

    logFile.close();
    cv::destroyAllWindows();
    return 0;
}

