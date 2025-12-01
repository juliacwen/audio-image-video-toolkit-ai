/******************************************************************************
 * File: video_block_matching.cpp
 * Description:
 *   Stereo disparity demo using classical block matching (StereoBM).
 *   Right image is simulated by shifting the left image horizontally.
 *   Disparity results reflect this artificial shift, NOT real depth.
 * Author: Julia Wen (wendigilane@gmail.com)
 * 09-05-2025 — Initial check-in  
 * 11-30-2025 — improvement
 ******************************************************************************/

#include "../video_common/inc/video_common.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace video_common;

namespace {
    constexpr int SHIFT_PIXELS = 4;  // artificial stereo baseline
    constexpr int OVERLAY_MARGIN_X = 10;
    constexpr int OVERLAY_MARGIN_Y = 10;
    constexpr double OVERLAY_FONT_SCALE = 0.5;
    constexpr int OVERLAY_THICKNESS = 1;
    constexpr int FPS_UPDATE_INTERVAL = 10;  // Update FPS calculation every N frames

    // Helper function to parse command-line arguments
    bool parseArguments(int argc, char** argv) {
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
                } catch (const std::invalid_argument& e) {
                    LOG(LogLevel::ERROR, "Invalid log level (not a number): " << argv[i + 1]);
                    return false;
                } catch (const std::out_of_range& e) {
                    LOG(LogLevel::ERROR, "Log level out of range: " << argv[i + 1]);
                    return false;
                }
            } else if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  -v, --verbose <level>  Set log level (1=INFO, 2=WARNING, 3=ERROR)\n"
                          << "  -h, --help             Show this help message\n";
                return false;
            } else {
                LOG(LogLevel::WARNING, "Unknown argument: " << arg);
            }
        }
        return true;
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

    // Helper function to create overlay text
    void addOverlay(cv::Mat& image, const std::string& timestamp, 
                   int frameCount, double fps) {
        std::string overlayText = timestamp + " | Frame: " + std::to_string(frameCount) +
                                  " | FPS: " + cv::format("%.2f", fps);
        cv::putText(image, overlayText,
                    cv::Point(OVERLAY_MARGIN_X, image.rows - OVERLAY_MARGIN_Y),
                    cv::FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE,
                    cv::Scalar(MAX_PIXEL_VALUE),
                    OVERLAY_THICKNESS, cv::LINE_AA);
    }
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    if (!parseArguments(argc, argv)) {
        return EXIT_FAILURE;
    }

    // Register Ctrl-C handler
    std::signal(SIGINT, handleSigInt);

    LOG(LogLevel::INFO, "=== BLOCK MATCHING INITIALIZED (StereoBM) ===");

    // Initialize camera
    cv::VideoCapture cap;
    if (!initCamera(cap, 0)) {
        return EXIT_FAILURE;
    }

    cv::Mat frame, gray, grayShifted;

    // Initialize StereoBM
    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(NUM_DISPARITIES, BLOCK_SIZE);
    
    // Validate shift amount is reasonable for block matching
    if (SHIFT_PIXELS >= NUM_DISPARITIES) {
        LOG(LogLevel::WARNING, "SHIFT_PIXELS (" << SHIFT_PIXELS 
            << ") should be less than NUM_DISPARITIES (" << NUM_DISPARITIES << ")");
    }
    if (SHIFT_PIXELS < BLOCK_SIZE / 2) {
        LOG(LogLevel::WARNING, "SHIFT_PIXELS (" << SHIFT_PIXELS 
            << ") is small relative to BLOCK_SIZE (" << BLOCK_SIZE << ")");
    }
    
    LOG(LogLevel::INFO, "Block size = " << BLOCK_SIZE
              << ", search range = " << NUM_DISPARITIES
              << " disparities. NOTE: Right image simulated by shifting left image by "
              << SHIFT_PIXELS << " pixels.");

    // Create windows
    cv::namedWindow("Original (Left)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Disparity Map", cv::WINDOW_AUTOSIZE);

    // Video writer setup
    cv::VideoWriter writer;
    bool writerInitialized = false;
    std::string outFile = "video_block_output.avi";

    // Frame tracking
    int frameCount = 0;
    int emptyFrameCount = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto lastFpsUpdate = startTime;
    double currentFps = 0.0;
    int framesSinceLastFpsUpdate = 0;

    while (!stopFlag.load(std::memory_order_acquire)) {
        cap >> frame;
        if (frame.empty()) {
            emptyFrameCount++;
            LOG(LogLevel::WARNING, "Empty frame captured (consecutive: " << emptyFrameCount << ")");
            
            if (emptyFrameCount >= MAX_EMPTY_FRAMES) {
                LOG(LogLevel::ERROR, "Too many consecutive empty frames. Exiting.");
                break;
            }
            continue;
        }
        emptyFrameCount = 0; // Reset counter

        ++frameCount;
        ++framesSinceLastFpsUpdate;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Update FPS calculation periodically for efficiency
        auto now = std::chrono::steady_clock::now();
        if (framesSinceLastFpsUpdate >= FPS_UPDATE_INTERVAL) {
            auto elapsed = now - lastFpsUpdate;
            currentFps = framesSinceLastFpsUpdate / std::chrono::duration<double>(elapsed).count();
            lastFpsUpdate = now;
            framesSinceLastFpsUpdate = 0;
        }

        // Simulate right frame with horizontal shift
        cv::Mat translationMatrix = (cv::Mat_<double>(2, 3) << 1, 0, SHIFT_PIXELS, 0, 1, 0);
        cv::warpAffine(gray, grayShifted, translationMatrix, gray.size());

        // Compute disparity
        cv::Mat disparity16S, disparity8U;
        stereoBM->compute(gray, grayShifted, disparity16S);
        
        // Normalize disparity for display
        // Using normalize is safer than manual scaling as it handles edge cases
        cv::normalize(disparity16S, disparity8U, 0, MAX_PIXEL_VALUE, cv::NORM_MINMAX, CV_8U);

        // Add overlay with current FPS
        addOverlay(disparity8U, getTimestamp(), frameCount, currentFps);

        // Initialize video writer on first successful frame
        if (!writerInitialized) {
            writer.open(outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                        VIDEO_FPS, disparity8U.size(), false);
            if (!writer.isOpened()) {
                LOG(LogLevel::ERROR, "Could not open " << outFile << " for writing");
                break;
            }
            writerInitialized = true;
            LOG(LogLevel::INFO, "VideoWriter initialized (" << outFile << ")");
        }

        // Write frame to video
        writer.write(disparity8U);

        // Log statistics periodically
        if (frameCount % LOG_INTERVAL == 0) {
            double minVal, maxVal;
            cv::minMaxLoc(disparity16S, &minVal, &maxVal);
            LOG(LogLevel::INFO, "[FRAME " << frameCount << "] "
                      << "Resolution=" << frame.cols << "x" << frame.rows
                      << " | FPS=" << cv::format("%.2f", currentFps)
                      << " | Disparity[min=" << minVal
                      << ", max=" << maxVal << "]");
        }

        // Display windows
        cv::imshow("Original (Left)", gray);
        cv::imshow("Disparity Map", disparity8U);

        // Check for exit keys
        int key = cv::waitKey(1);
        if (key == KEY_EXIT_ESC || key == KEY_EXIT_CTRL_C) {
            LOG(LogLevel::INFO, "Exit key pressed. Stopping...");
            break;
        }
    }

    // Cleanup
    cap.release();
    writer.release();
    cv::destroyAllWindows();
    
    LOG(LogLevel::INFO, "Finished. Video saved to " << outFile 
                        << " (total frames: " << frameCount << ")");

    return EXIT_SUCCESS;
}