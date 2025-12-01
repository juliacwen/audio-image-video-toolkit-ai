/******************************************************************************
 * File: video_motion_estimation.cpp
 * Description: Capture from camera, compute dense optical flow (Farneback),
 *              display motion vectors, and write frames to MJPG/AVI file.
 *              Logging via -v option and Ctrl-C support.
 * Author: Julia Wen (wendigilane@gmail.com)
 * 09-05-2025 — Initial check-in  
 * 11-30-2025 — improvement
 ******************************************************************************/

#include "../video_common/inc/video_common.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace video_common;

namespace {
    // Helper function to initialize camera
    bool initCamera(cv::VideoCapture& cap, int deviceId = 0) {
        cap.open(deviceId);
        if (!cap.isOpened()) {
            LOG(LogLevel::ERROR, "Cannot open camera with device ID: " << deviceId);
            return false;
        }
        LOG(LogLevel::INFO, "Camera opened successfully (device " << deviceId << ")");
        return true;
    }

    // Helper function to initialize video writer
    bool initVideoWriter(cv::VideoWriter& writer, const std::string& filename,
                        int fps, const cv::Size& frameSize) {
        writer.open(filename,
                   cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                   fps,
                   frameSize);
        if (!writer.isOpened()) {
            LOG(LogLevel::ERROR, "Could not open video file for writing: " << filename);
            return false;
        }
        LOG(LogLevel::INFO, "Writing to: " << filename);
        return true;
    }

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

    // Helper function to draw motion vectors
    void drawMotionVectors(cv::Mat& display, const cv::Mat& flow) {
        for (int y = 0; y < display.rows; y += DISPLAY_STEP) {
            for (int x = 0; x < display.cols; x += DISPLAY_STEP) {
                const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
                
                // Use squared magnitude for performance
                float magSquared = fxy.x * fxy.x + fxy.y * fxy.y;
                cv::Scalar color = (magSquared > HIGHLIGHT_THRESHOLD_SQ) ? STRONG_COLOR : NORMAL_COLOR;
                
                cv::line(display, cv::Point(x, y),
                        cv::Point(cvRound(x + fxy.x * VECTOR_SCALE),
                                 cvRound(y + fxy.y * VECTOR_SCALE)),
                        color, 2);
                cv::circle(display, cv::Point(x, y), REF_POINT_RADIUS, REF_COLOR, -1);
            }
        }
    }

    // Helper function to log debug points
    void logDebugPoints(const cv::Mat& flow, int frameCount) {
        for (const auto& pt : DEBUG_POINTS) {
            // Validate bounds before accessing
            if (isPointValid(pt, flow.rows, flow.cols)) {
                const cv::Point2f& f = flow.at<cv::Point2f>(pt.y, pt.x);
                LOG(LogLevel::INFO, "Frame " << frameCount
                                    << " motion(" << pt.x << "," << pt.y
                                    << "): dx=" << f.x << ", dy=" << f.y);
            } else {
                LOG(LogLevel::WARNING, "Debug point (" << pt.x << "," << pt.y
                                       << ") is out of bounds for frame size "
                                       << flow.cols << "x" << flow.rows);
            }
        }
    }
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    if (!parseArguments(argc, argv)) {
        return EXIT_FAILURE;
    }

    // Register Ctrl-C handler
    std::signal(SIGINT, handleSigInt);

    LOG(LogLevel::INFO, "=== Video Motion Estimation ===");

    // Initialize camera
    cv::VideoCapture cap;
    if (!initCamera(cap, 0)) {
        LOG(LogLevel::ERROR, "Failed to initialize camera. Make sure:");
        LOG(LogLevel::ERROR, "  1. Camera is connected");
        LOG(LogLevel::ERROR, "  2. No other app is using the camera");
        LOG(LogLevel::ERROR, "  3. Terminal has camera permissions (System Preferences > Security & Privacy > Camera)");
        return EXIT_FAILURE;
    }
    
    // Log camera properties
    LOG(LogLevel::INFO, "Camera resolution: " 
        << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x" 
        << cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    LOG(LogLevel::INFO, "Camera FPS: " << cap.get(cv::CAP_PROP_FPS));

    // Capture first frame
    cv::Mat prevFrame, prevGray;
    cap >> prevFrame;
    if (prevFrame.empty()) {
        LOG(LogLevel::ERROR, "Empty first frame captured.");
        return EXIT_FAILURE;
    }
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

    // Initialize video writer
    std::string outFile = "output_video.avi";
    cv::VideoWriter writer;
    if (!initVideoWriter(writer, outFile, VIDEO_FPS, prevFrame.size())) {
        return EXIT_FAILURE;
    }

    // Main processing loop
    cv::Mat currFrame, currGray;
    int frameCount = 0;
    int emptyFrameCount = 0;

    while (!stopFlag.load(std::memory_order_acquire)) {
        cap >> currFrame;
        if (currFrame.empty()) {
            emptyFrameCount++;
            LOG(LogLevel::WARNING, "Empty frame captured at frame " << frameCount
                                   << " (consecutive: " << emptyFrameCount << ")");
            
            if (emptyFrameCount >= MAX_EMPTY_FRAMES) {
                LOG(LogLevel::ERROR, "Too many consecutive empty frames. Exiting.");
                break;
            }
            continue;
        }
        emptyFrameCount = 0; // Reset counter on successful frame

        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

        // Compute dense optical flow
        cv::Mat flow = computeVideoMotionField(prevGray, currGray);

        // Create display frame
        cv::Mat display = currFrame.clone();
        
        // Draw motion vectors
        drawMotionVectors(display, flow);

        // Log debug points
        logDebugPoints(flow, frameCount);

        // Display windows
        cv::imshow("Motion Vectors Full", display);
        
        // Create and display scaled version
        cv::Mat displayScaled;
        cv::resize(display, displayScaled, cv::Size(), WINDOW_SCALE, WINDOW_SCALE);
        cv::imshow("Motion Vectors Scaled", displayScaled);

        // Write frame to video
        writer.write(display);

        // Check for exit keys
        int key = cv::waitKey(1);
        if (key == KEY_EXIT_CTRL_C || key == KEY_EXIT_ESC) {
            LOG(LogLevel::INFO, "Exit key pressed. Stopping capture.");
            break;
        }

        // Update previous frame (swap instead of clone for efficiency)
        std::swap(prevGray, currGray);
        frameCount++;
    }

    // Cleanup
    cap.release();
    writer.release();
    cv::destroyAllWindows();
    
    LOG(LogLevel::INFO, "Video capture and writing completed: " << outFile
                        << " (total frames: " << frameCount << ")");

    return EXIT_SUCCESS;
}