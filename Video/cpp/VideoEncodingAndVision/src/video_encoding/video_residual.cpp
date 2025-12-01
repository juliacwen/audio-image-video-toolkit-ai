/******************************************************************************
 * File: video_residual.cpp
 * Description: Compute residual between current and predicted frames.
 *              - Motion estimation
 *              - Timestamp overlay
 *              - Two windows: Predicted + Residual
 *              - Logging with -v
 *              - Exit on ESC or Ctrl-C
 * Author: Julia Wen (wendigilane@gmail.com)
 * 09-05-2025 — Initial check-in  
 * 11-30-2025 — improvement
 ******************************************************************************/

#include <opencv2/opencv.hpp>
#include "../video_common/inc/video_common.h"
#include <iostream>
#include <cmath>
#include <csignal>
#include <cstdlib>

using namespace video_common;

namespace {
    // Local visual constants
    constexpr int ARROW_THICKNESS = 1;
    constexpr double FONT_SCALE = 0.5;
    constexpr int FONT_THICKNESS = 1;
    constexpr int OVERLAY_MARGIN_X = 10;
    constexpr int OVERLAY_MARGIN_Y = 10;

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

    // Helper to draw motion vectors and compute max displacement
    float drawMotionVectors(cv::Mat& displayPred, const cv::Mat& flow, int frameIdx) {
        float maxDisp = 0.0f;
        int largeMotionCount = 0;

        for (int y = 0; y < flow.rows; y += SAMPLE_STEP) {
            for (int x = 0; x < flow.cols; x += SAMPLE_STEP) {
                if (!isPointValid(cv::Point(x, y), flow.rows, flow.cols)) {
                    continue;
                }

                const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
                float magSquared = fxy.x * fxy.x + fxy.y * fxy.y;
                
                // Update max using squared values until final result
                if (magSquared > maxDisp * maxDisp) {
                    maxDisp = std::sqrt(magSquared);
                }

                // Draw arrow
                cv::Point p0(x, y);
                cv::Point p1(
                    cv::saturate_cast<int>(x + fxy.x * VECTOR_SCALE),
                    cv::saturate_cast<int>(y + fxy.y * VECTOR_SCALE)
                );
                cv::arrowedLine(displayPred, p0, p1, STRONG_COLOR, ARROW_THICKNESS, cv::LINE_AA);
                cv::circle(displayPred, p0, REF_POINT_RADIUS, REF_COLOR, -1);

                // Only log large motions to avoid flooding logs
                if (magSquared > HIGHLIGHT_THRESHOLD_SQ) {
                    largeMotionCount++;
                    if (largeMotionCount <= 5) { // Limit to first 5 per frame
                        float mag = std::sqrt(magSquared);
                        LOG(LogLevel::INFO, "Frame " << frameIdx << " large motion at (" 
                            << x << "," << y << "): magnitude=" << mag);
                    }
                }
            }
        }

        if (largeMotionCount > 5) {
            LOG(LogLevel::INFO, "Frame " << frameIdx << ": " << (largeMotionCount - 5) 
                << " more large motions not shown");
        }

        return maxDisp;
    }

    // Helper to add overlay text
    void addOverlay(cv::Mat& image, const std::string& timestamp, int frameIdx, float maxDisp) {
        std::string overlayText = timestamp +
                                  " | Frame: " + std::to_string(frameIdx) +
                                  " | MaxDisp: " + cv::format("%.2f", maxDisp);
        cv::putText(image, overlayText,
                   cv::Point(OVERLAY_MARGIN_X, image.rows - OVERLAY_MARGIN_Y),
                   cv::FONT_HERSHEY_SIMPLEX,
                   FONT_SCALE,
                   cv::Scalar(MAX_PIXEL_VALUE, MAX_PIXEL_VALUE, MAX_PIXEL_VALUE),
                   FONT_THICKNESS, cv::LINE_AA);
    }
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    if (!parseArguments(argc, argv)) {
        return EXIT_FAILURE;
    }

    // Register Ctrl-C handler
    std::signal(SIGINT, handleSigInt);

    LOG(LogLevel::INFO, "=== Video Residual Computation ===");
    LOG(LogLevel::INFO, "Logging level: " << static_cast<int>(logLevel));

    // Initialize camera
    cv::VideoCapture cap;
    if (!initCamera(cap, 0)) {
        return EXIT_FAILURE;
    }

    // Capture first frame
    cv::Mat prevFrame, prevGray;
    cap >> prevFrame;
    if (prevFrame.empty()) {
        LOG(LogLevel::ERROR, "Empty frame captured at start.");
        return EXIT_FAILURE;
    }
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

    // Initialize video writers
    const std::string outPredFile = "predicted_output.avi";
    const std::string outResFile  = "residual_output.avi";
    
    cv::VideoWriter writerPred(outPredFile,
                              cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                              VIDEO_FPS,
                              prevFrame.size(), /*isColor=*/true);
    cv::VideoWriter writerRes(outResFile,
                             cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                             VIDEO_FPS,
                             prevFrame.size(), /*isColor=*/false);

    if (!writerPred.isOpened() || !writerRes.isOpened()) {
        LOG(LogLevel::ERROR, "Could not open video writers.");
        return EXIT_FAILURE;
    }
    LOG(LogLevel::INFO, "Video writers initialized:");
    LOG(LogLevel::INFO, "  Predicted: " << outPredFile);
    LOG(LogLevel::INFO, "  Residual: " << outResFile);

    // Main processing loop
    int frameIdx = 0;
    int emptyFrameCount = 0;
    cv::Mat currFrame, currGray;

    while (!stopFlag.load(std::memory_order_acquire)) {
        cap >> currFrame;
        if (currFrame.empty()) {
            emptyFrameCount++;
            LOG(LogLevel::WARNING, "Empty frame at index " << frameIdx 
                                   << " (consecutive: " << emptyFrameCount << ")");
            
            if (emptyFrameCount >= MAX_EMPTY_FRAMES) {
                LOG(LogLevel::ERROR, "Too many consecutive empty frames. Exiting.");
                break;
            }
            continue;
        }
        emptyFrameCount = 0; // Reset counter

        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

        // Compute motion field
        cv::Mat flow = computeVideoMotionField(prevGray, currGray);

        // Predict frame using motion-compensated flow
        cv::Mat predicted = predictNextVideoFrame(prevGray, flow);

        // Compute residual between current and predicted
        cv::Mat residual = computeVideoResidual(currGray, predicted);

        // Overlay motion vectors on predicted frame
        cv::Mat displayPred;
        cv::cvtColor(predicted, displayPred, cv::COLOR_GRAY2BGR);
        
        float maxDisp = drawMotionVectors(displayPred, flow, frameIdx);

        // Add overlay text
        addOverlay(displayPred, getTimestamp(), frameIdx, maxDisp);

        // Show windows
        cv::imshow("Predicted Frame (overlay)", displayPred);
        cv::imshow("Residual Frame", residual);

        // Write videos
        writerPred.write(displayPred);
        writerRes.write(residual);

        // Update previous frame (swap for efficiency)
        std::swap(prevGray, currGray);
        ++frameIdx;

        // Check for exit keys
        int key = cv::waitKey(1);
        if (key == KEY_EXIT_ESC || key == KEY_EXIT_CTRL_C) {
            LOG(LogLevel::INFO, "Exit key pressed. Stopping...");
            break;
        }
    }

    // Cleanup
    cap.release();
    writerPred.release();
    writerRes.release();
    cv::destroyAllWindows();

    LOG(LogLevel::INFO, "Predicted frames written to " << outPredFile 
                        << " (total frames: " << frameIdx << ")");
    LOG(LogLevel::INFO, "Residual frames written to " << outResFile 
                        << " (total frames: " << frameIdx << ")");

    return EXIT_SUCCESS;
}
