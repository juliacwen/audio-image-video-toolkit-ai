/******************************************************************************
 * File: video_depth_from_stereo.cpp
 * Description: Stereo depth demo with logging, grayscale + colorized disparity.
 *              - Simulates right camera if only one camera is available (pixel shift)
 *              - Logs per-frame metrics and run summary
 *              - Video outputs: grayscale disparity, colorized disparity, left & right views
 *              - Fully modernized for C++17; file-local constants kept in .cpp
 * Return: EXIT_SUCCESS (0)  -> success
 *         EXIT_FAILURE (1)  -> error (e.g., camera or log file could not be opened)
 * Author: Julia Wen (wendigilane@gmail.com)
 * 09-05-2025 — Initial check-in  
 * 12-01-2025 — improvement
 ******************************************************************************/

#include <opencv2/opencv.hpp>
#include "../video_common/inc/video_common.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <csignal>
#include <cstdlib>

using namespace video_common;

namespace {
    // =================== Constants (file-local) ===================
    constexpr int DEFAULT_SHIFT_PIXELS = 5;
    constexpr int MIN_SHIFT_PIXELS = 1;
    constexpr int MAX_SHIFT_PIXELS = 100;
    constexpr double DISPARITY_SCALE = 255.0;
    const int VIDEO_CODEC = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    // =============================================================

    // Output filenames
    const std::string DEPTH_FILE        = "depth_output.avi";
    const std::string DEPTH_COLOR_FILE  = "depth_color_output.avi";
    const std::string LEFT_FILE         = "left_output.avi";
    const std::string RIGHT_FILE        = "right_output.avi";
    const std::string LOG_FILE          = "video_depth_log.txt";

    // Helper: normalize and colorize disparity
    cv::Mat normalizeAndColorizeDisparity(const cv::Mat& disparity) {
        if (disparity.empty()) {
            LOG(LogLevel::WARNING, "Empty disparity map for colorization");
            return cv::Mat();
        }

        double minVal, maxVal;
        cv::minMaxLoc(disparity, &minVal, &maxVal);
        
        if (maxVal - minVal < 1e-6) {
            LOG(LogLevel::WARNING, "Disparity range too small for normalization");
            return cv::Mat::zeros(disparity.size(), CV_8UC3);
        }

        cv::Mat normalizedDisp, colorDisp;
        disparity.convertTo(normalizedDisp, CV_8U, DISPARITY_SCALE / (maxVal - minVal),
                           -minVal * DISPARITY_SCALE / (maxVal - minVal));
        cv::applyColorMap(normalizedDisp, colorDisp, cv::COLORMAP_JET);
        return colorDisp;
    }

    // Helper function to parse command-line arguments
    struct Config {
        int shiftPixels = DEFAULT_SHIFT_PIXELS;
        bool showHelp = false;
    };

    bool parseArguments(int argc, char** argv, Config& config) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "-v" || arg == "--verbose") {
                if (i + 1 >= argc) {
                    LOG(LogLevel::ERROR, "Missing value for -v option");
                    return false;
                }
                
                try {
                    int lvl = std::stoi(argv[i + 1]);
                    if (lvl >= static_cast<int>(LogLevel::INFO) &&
                        lvl <= static_cast<int>(LogLevel::ERROR)) {
                        logLevel = static_cast<LogLevel>(lvl);
                        ++i; // Skip the next argument
                    } else {
                        LOG(LogLevel::ERROR, "Log level must be between "
                            << static_cast<int>(LogLevel::INFO) << " and "
                            << static_cast<int>(LogLevel::ERROR));
                        return false;
                    }
                } catch (const std::exception& e) {
                    LOG(LogLevel::ERROR, "Invalid log level: " << argv[i + 1]);
                    return false;
                }
            } else if (arg == "-s" || arg == "--shift") {
                if (i + 1 >= argc) {
                    LOG(LogLevel::ERROR, "Missing value for -s option");
                    return false;
                }
                
                try {
                    config.shiftPixels = std::stoi(argv[i + 1]);
                    if (config.shiftPixels < MIN_SHIFT_PIXELS || 
                        config.shiftPixels > MAX_SHIFT_PIXELS) {
                        LOG(LogLevel::ERROR, "Shift must be between " 
                            << MIN_SHIFT_PIXELS << " and " << MAX_SHIFT_PIXELS);
                        return false;
                    }
                    ++i; // Skip the next argument
                } catch (const std::exception& e) {
                    LOG(LogLevel::ERROR, "Invalid shift value: " << argv[i + 1]);
                    return false;
                }
            } else if (arg == "-h" || arg == "--help") {
                config.showHelp = true;
                return true;
            } else {
                LOG(LogLevel::WARNING, "Unknown argument: " << arg);
            }
        }
        return true;
    }

    void showHelp(const char* programName) {
        std::cout << "Usage: " << programName << " [options]\n"
                  << "Options:\n"
                  << "  -v, --verbose <level>     Set log level (1=INFO, 2=WARNING, 3=ERROR)\n"
                  << "  -s, --shift <pixels>      Horizontal shift for simulated stereo (" 
                  << MIN_SHIFT_PIXELS << "-" << MAX_SHIFT_PIXELS 
                  << ", default=" << DEFAULT_SHIFT_PIXELS << ")\n"
                  << "  -h, --help                Show this help message\n"
                  << "\nOutputs:\n"
                  << "  " << DEPTH_FILE << " - Grayscale disparity\n"
                  << "  " << DEPTH_COLOR_FILE << " - Colorized disparity\n"
                  << "  " << LEFT_FILE << " - Left camera view\n"
                  << "  " << RIGHT_FILE << " - Simulated right view\n"
                  << "  " << LOG_FILE << " - Per-frame statistics\n";
    }

    // Helper function to initialize camera
    bool initCamera(cv::VideoCapture& cap, int deviceId = 0) {
        cap.open(deviceId);
        if (!cap.isOpened()) {
            LOG(LogLevel::ERROR, "Cannot open camera with device ID: " << deviceId);
            return false;
        }
        LOG(LogLevel::INFO, "Camera opened successfully (device " << deviceId << ")");
        LOG(LogLevel::INFO, "Camera resolution: " 
            << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x" 
            << cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        LOG(LogLevel::INFO, "Camera FPS: " << cap.get(cv::CAP_PROP_FPS));
        return true;
    }
}

// =================== Main ===================
int main(int argc, char** argv) {
    // Parse arguments
    Config config;
    if (!parseArguments(argc, argv, config)) {
        return EXIT_FAILURE;
    }

    if (config.showHelp) {
        showHelp(argv[0]);
        return EXIT_SUCCESS;
    }

    // Register Ctrl-C handler
    std::signal(SIGINT, handleSigInt);

    LOG(LogLevel::INFO, "=== Stereo Depth from Simulated Stereo ===");

    // Open left camera
    cv::VideoCapture capLeft;
    if (!initCamera(capLeft, 0)) {
        return EXIT_FAILURE;
    }

    int camWidth  = static_cast<int>(capLeft.get(cv::CAP_PROP_FRAME_WIDTH));
    int camHeight = static_cast<int>(capLeft.get(cv::CAP_PROP_FRAME_HEIGHT));
    double camFps = capLeft.get(cv::CAP_PROP_FPS);
    if (camFps <= 0) {
        camFps = VIDEO_FPS; // Use constant from video_common
        LOG(LogLevel::WARNING, "Invalid camera FPS, using default: " << camFps);
    }

    // Open log file
    std::ofstream logFile(LOG_FILE);
    if (!logFile.is_open()) {
        LOG(LogLevel::ERROR, "Cannot open log file: " << LOG_FILE);
        return EXIT_FAILURE;
    }

    auto runStart = std::chrono::steady_clock::now();

    // Write header to log file
    logFile << "===== Stereo Depth Run =====" << std::endl;
    logFile << "Start Time : " << getTimestamp() << std::endl;
    logFile << "Shift      : " << config.shiftPixels << " px" << std::endl;
    logFile << "Log Level  : " << static_cast<int>(logLevel) << std::endl;
    logFile << "Resolution : " << camWidth << "x" << camHeight << std::endl;
    logFile << "Camera FPS : " << camFps << std::endl;
    logFile << "Output     : " << DEPTH_FILE       << " (grayscale disparity)" << std::endl;
    logFile << "           : " << DEPTH_COLOR_FILE << " (colorized disparity)" << std::endl;
    logFile << "           : " << LEFT_FILE        << " (left camera)" << std::endl;
    logFile << "           : " << RIGHT_FILE       << " (simulated right)" << std::endl;
    logFile << "============================" << std::endl;

    cv::Mat leftFrame, rightFrame, leftGray, rightGray, disparity, disparityColor;
    cv::VideoWriter writerDisp, writerLeft, writerRight, writerDispColor;
    bool writerInitialized = false;

    long frameIndex = 0;
    int emptyFrameCount = 0;
    double sumDisparity = 0.0;

    while (!stopFlag.load(std::memory_order_acquire)) {
        capLeft >> leftFrame;
        if (leftFrame.empty()) {
            emptyFrameCount++;
            LOG(LogLevel::WARNING, "Empty frame (consecutive: " << emptyFrameCount << ")");
            
            if (emptyFrameCount >= MAX_EMPTY_FRAMES) {
                LOG(LogLevel::ERROR, "Too many consecutive empty frames. Exiting.");
                break;
            }
            continue;
        }
        emptyFrameCount = 0; // Reset counter
        frameIndex++;

        // Simulate right camera with horizontal shift
        cv::Mat shiftMat = (cv::Mat_<double>(2, 3) << 1, 0, config.shiftPixels, 0, 1, 0);
        cv::warpAffine(leftFrame, rightFrame, shiftMat, leftFrame.size());

        // Grayscale conversion
        cv::cvtColor(leftFrame, leftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightFrame, rightGray, cv::COLOR_BGR2GRAY);

        // Compute disparity
        disparity = computeVideoDisparity(leftGray, rightGray);

        // Colorize disparity
        disparityColor = normalizeAndColorizeDisparity(disparity);
        if (disparityColor.empty()) {
            LOG(LogLevel::WARNING, "Frame " << frameIndex << ": Failed to colorize disparity");
            continue;
        }

        // Initialize video writers once we have valid frames
        if (!writerInitialized) {
            writerDisp.open(DEPTH_FILE, VIDEO_CODEC, camFps, disparity.size(), false);
            writerDispColor.open(DEPTH_COLOR_FILE, VIDEO_CODEC, camFps, disparityColor.size(), true);
            writerLeft.open(LEFT_FILE, VIDEO_CODEC, camFps, leftFrame.size(), true);
            writerRight.open(RIGHT_FILE, VIDEO_CODEC, camFps, rightFrame.size(), true);

            if (!writerDisp.isOpened() || !writerDispColor.isOpened() ||
                !writerLeft.isOpened() || !writerRight.isOpened()) {
                LOG(LogLevel::ERROR, "Cannot open one of the video writers");
                return EXIT_FAILURE;
            }
            writerInitialized = true;
            LOG(LogLevel::INFO, "Video writers initialized successfully");
        }

        // Write videos
        writerDisp.write(disparity);
        writerDispColor.write(disparityColor);
        writerLeft.write(leftFrame);
        writerRight.write(rightFrame);

        // Log per-frame statistics
        cv::Scalar meanDisp = cv::mean(disparity);
        sumDisparity += meanDisp[0];
        
        if (frameIndex % LOG_INTERVAL == 0) {
            std::string ts = getTimestamp();
            logFile << "Frame " << frameIndex << " @ [" << ts 
                   << "] avg disparity: " << meanDisp[0] << std::endl;
            LOG(LogLevel::INFO, "Frame " << frameIndex << " avg disparity: " << meanDisp[0]);
        }

        // Display
        cv::imshow("Depth / Disparity Map (Colorized)", disparityColor);
        
        int key = cv::waitKey(1);
        if (key == KEY_EXIT_ESC || key == KEY_EXIT_CTRL_C) {
            LOG(LogLevel::INFO, "Exit key pressed. Stopping...");
            break;
        }
    }

    // Cleanup
    capLeft.release();
    writerDisp.release();
    writerDispColor.release();
    writerLeft.release();
    writerRight.release();

    // Calculate summary statistics
    auto runEnd = std::chrono::steady_clock::now();
    double durationSec = std::chrono::duration_cast<std::chrono::milliseconds>(runEnd - runStart).count() / 1000.0;
    double avgDisparity = (frameIndex > 0) ? sumDisparity / frameIndex : 0.0;

    // Write summary to log file
    logFile << "===== Run Summary =====" << std::endl;
    logFile << "Total Frames Captured : " << frameIndex << std::endl;
    logFile << "Run Duration          : " << durationSec << " seconds" << std::endl;
    logFile << "Average Disparity     : " << avgDisparity << std::endl;
    logFile << "=======================" << std::endl;

    logFile.close();
    cv::destroyAllWindows();

    LOG(LogLevel::INFO, "Processing complete. Total frames: " << frameIndex);
    LOG(LogLevel::INFO, "Videos saved to: " << DEPTH_FILE << ", " << DEPTH_COLOR_FILE 
                        << ", " << LEFT_FILE << ", " << RIGHT_FILE);
    LOG(LogLevel::INFO, "Statistics saved to: " << LOG_FILE);

    return EXIT_SUCCESS;
}