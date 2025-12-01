/******************************************************************************
 * File: video_frame_prediction.cpp
 * Description:
 *   Demo program to perform motion-compensated frame prediction using dense
 *   optical flow. Uses constants and LOG from video_common.
 * Author: Julia Wen (wendigilane@gmail.com)
 * 09-05-2025 — Initial check-in  
 * 11-30-2025 — improvement
 ******************************************************************************/


#include "../video_common/inc/video_common.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>

using namespace video_common;

namespace {
    constexpr int OVERLAY_MARGIN_X = 10;
    constexpr int OVERLAY_MARGIN_Y = 10;
    constexpr double OVERLAY_FONT_SCALE = 0.5;
    constexpr int OVERLAY_THICKNESS = 1;

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

    // Helper to compute motion statistics
    struct MotionStats {
        float maxDisplacement;
        int largeMotionCount;
    };

    MotionStats analyzeMotion(const cv::Mat& flow, int frameIdx) {
        MotionStats stats{0.0f, 0};
        
        for (int y = 0; y < flow.rows; y += SAMPLE_STEP) {
            for (int x = 0; x < flow.cols; x += SAMPLE_STEP) {
                if (!isPointValid(cv::Point(x, y), flow.rows, flow.cols)) {
                    continue;
                }
                
                const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
                float magSquared = fxy.x * fxy.x + fxy.y * fxy.y;
                
                // Update max displacement (use squared until final value)
                if (magSquared > stats.maxDisplacement * stats.maxDisplacement) {
                    stats.maxDisplacement = std::sqrt(magSquared);
                }
                
                // Check for large motion using squared threshold
                if (magSquared > HIGHLIGHT_THRESHOLD_SQ) {
                    stats.largeMotionCount++;
                    float mag = std::sqrt(magSquared);
                    LOG(LogLevel::WARNING, "Large motion at frame " << frameIdx
                        << " (" << x << "," << y << ") = " << mag);
                }
            }
        }
        
        return stats;
    }

    // Helper to draw motion vectors
    void drawMotionVectors(cv::Mat& image, const cv::Mat& flow) {
        for (int y = 0; y < flow.rows; y += SAMPLE_STEP) {
            for (int x = 0; x < flow.cols; x += SAMPLE_STEP) {
                if (!isPointValid(cv::Point(x, y), flow.rows, flow.cols)) {
                    continue;
                }
                
                const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
                cv::circle(image, cv::Point(x, y), REF_POINT_RADIUS, REF_COLOR, -1);
                cv::arrowedLine(image,
                               cv::Point(x, y),
                               cv::Point(cvRound(x + fxy.x * VECTOR_SCALE), 
                                       cvRound(y + fxy.y * VECTOR_SCALE)),
                               STRONG_COLOR, 1, cv::LINE_AA);
            }
        }
    }

    // Helper to add overlay text
    void addOverlay(cv::Mat& image, int frameIdx, double mad, float maxDisp) {
        std::string overlayText = "Frame: " + std::to_string(frameIdx) +
                                  " | MAD: " + cv::format("%.2f", mad) +
                                  " | MaxDisp: " + cv::format("%.2f", maxDisp);
        cv::putText(image, overlayText,
                   cv::Point(OVERLAY_MARGIN_X, image.rows - OVERLAY_MARGIN_Y),
                   cv::FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE,
                   cv::Scalar(MAX_PIXEL_VALUE, MAX_PIXEL_VALUE, MAX_PIXEL_VALUE),
                   OVERLAY_THICKNESS, cv::LINE_AA);
    }
}

// Main program
int main(int argc, char** argv) {
    // Parse command-line arguments
    if (!parseArguments(argc, argv)) {
        return EXIT_FAILURE;
    }

    // Register Ctrl-C handler
    std::signal(SIGINT, handleSigInt);

    LOG(LogLevel::INFO, "=== Video Frame Prediction ===");

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

    // Initialize video writer
    std::string outFile = "predicted_output.avi";
    cv::VideoWriter writer(outFile,
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          VIDEO_FPS,
                          prevFrame.size());
    if (!writer.isOpened()) {
        LOG(LogLevel::ERROR, "Could not open video file for writing: " << outFile);
        return EXIT_FAILURE;
    }
    LOG(LogLevel::INFO, "VideoWriter initialized: " << outFile);

    // Main processing loop
    cv::Mat currFrame, currGray;
    int frameIdx = 0;
    int emptyFrameCount = 0;

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

        // Motion-compensated predicted frame
        cv::Mat predicted = predictNextVideoFrame(prevGray, flow);

        // Compute mean absolute difference
        cv::Mat diff;
        cv::absdiff(predicted, currGray, diff);
        double mad = cv::mean(diff)[0];

        // Analyze motion statistics
        MotionStats stats = analyzeMotion(flow, frameIdx);

        // Convert predicted to BGR for overlay
        cv::Mat predictedBGR;
        cv::cvtColor(predicted, predictedBGR, cv::COLOR_GRAY2BGR);

        // Draw motion vectors
        drawMotionVectors(predictedBGR, flow);

        // Add overlay text
        addOverlay(predictedBGR, frameIdx, mad, stats.maxDisplacement);

        // Display windows
        cv::imshow("Predicted Frame", predictedBGR);
        
        cv::Mat predictedSmall;
        cv::resize(predictedBGR, predictedSmall, cv::Size(), WINDOW_SCALE, WINDOW_SCALE);
        cv::imshow("Predicted Frame Scaled", predictedSmall);

        // Write video
        writer.write(predictedBGR);

        // Update previous frame (swap for efficiency)
        std::swap(prevGray, currGray);
        frameIdx++;

        // Check for exit keys
        int key = cv::waitKey(1);
        if (key == KEY_EXIT_CTRL_C || key == KEY_EXIT_ESC) {
            LOG(LogLevel::INFO, "Exit key pressed. Stopping...");
            break;
        }
    }

    // Cleanup
    cap.release();
    writer.release();
    cv::destroyAllWindows();
    
    LOG(LogLevel::INFO, "Predicted frames written to " << outFile 
                        << " (total frames: " << frameIdx << ")");
    
    return EXIT_SUCCESS;
}