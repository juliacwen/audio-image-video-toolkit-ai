/******************************************************************************
 * File: video_vio_demo.cpp
 * Description: Simplified VIO demo combining motion vectors and depth (disparity)
 *              to visualize camera trajectory in 2D. Supports a single camera
 *              by simulating the right frame with a horizontal shift.
 *              Logs per-frame metrics (position, flow, depth) to CSV, trace,
 *              and combined log files.
 * Return: EXIT_SUCCESS (0)  -> success
 *         EXIT_FAILURE (1)  -> error (cannot open camera or log files)
 * Author: Julia Wen (wendigilane@gmail.com)
 * 09-05-2025 — Initial check-in  
 * 12-01-2025 — improvement
 ******************************************************************************/

#include <opencv2/opencv.hpp>
#include "../video_common/inc/video_common.h"
#include <iostream>
#include <fstream>
#include <deque>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <string>
#include <csignal>
#include <cstdlib>

using namespace video_common;

namespace {
    // =================== Constants (file-local) ===================
    constexpr int TRAJ_WIDTH         = 800;
    constexpr int TRAJ_HEIGHT        = 600;
    constexpr int TRAJ_CIRCLE_RADIUS = 2;
    constexpr int SHIFT_PIXELS       = 15;       // simulated right camera shift
    constexpr int TRAJ_DISPLAY_SCALE = 4;        // downscale trajectory overlay
    constexpr float FLOW_SCALE       = 20.0f;    // amplify flow for visible trajectory
    constexpr float DEPTH_MAX_VALUE  = 255.0f;   // maximum depth for scaling
    constexpr int FLOW_ARROW_STEP    = 10;       // step size for drawing flow arrows
    constexpr int OVERLAY_MARGIN     = 10;       // margin for trajectory overlay
    
    const cv::Point2f START_POS(TRAJ_WIDTH / 2.0f, TRAJ_HEIGHT / 2.0f);
    const cv::Scalar TRAJ_COLOR(0, 255, 0);
    const cv::Scalar FLOW_ARROW_COLOR(0, 0, 255);
    const std::string OUTPUT_DIR = "vio_output";
    // ==================================================

    struct Options {
        bool showHelp = false;
    };

    bool parseArguments(int argc, char** argv, Options& opt) {
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
            } else if (arg == "-h" || arg == "--help") {
                opt.showHelp = true;
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
                  << "  -h, --help                Show this help message\n"
                  << "\nDescription:\n"
                  << "  Simplified VIO demo that combines optical flow and disparity to\n"
                  << "  estimate and visualize camera trajectory. Uses simulated stereo\n"
                  << "  (single camera with horizontal shift).\n"
                  << "\nControls:\n"
                  << "  ESC - Exit program\n"
                  << "\nOutputs (in " << OUTPUT_DIR << "/):\n"
                  << "  motion_log_*.csv          - Frame-by-frame position and flow data\n"
                  << "  trace_log_*.txt           - Detailed trace log\n"
                  << "  combined_log_*.log        - Combined CSV and trace\n"
                  << "  vio_trajectory_*.avi      - Video with trajectory overlay\n";
    }

    bool initializeOutputDirectory() {
        try {
            std::filesystem::create_directories(OUTPUT_DIR);
            return true;
        } catch (const std::filesystem::filesystem_error& e) {
            LOG(LogLevel::ERROR, "Failed to create output directory: " << e.what());
            return false;
        }
    }

    bool openLogFiles(const std::string& ts, std::ofstream& csvFile,
                     std::ofstream& traceFile, std::ofstream& combFile) {
        csvFile.open(OUTPUT_DIR + "/motion_log_" + ts + ".csv");
        traceFile.open(OUTPUT_DIR + "/trace_log_" + ts + ".txt");
        combFile.open(OUTPUT_DIR + "/combined_log_" + ts + ".log");
        
        if (!csvFile.is_open() || !traceFile.is_open() || !combFile.is_open()) {
            LOG(LogLevel::ERROR, "Cannot open log files");
            return false;
        }
        
        csvFile << "frame,timestamp_ms,pos_x,pos_y,avg_flow_x,avg_flow_y,avg_depth\n";
        combFile << "CSV_HEADER: frame,timestamp_ms,pos_x,pos_y,avg_flow_x,avg_flow_y,avg_depth\n";
        
        return true;
    }

    void drawFlowArrows(cv::Mat& display, const cv::Mat& flow) {
        for (int y = 0; y < flow.rows; y += FLOW_ARROW_STEP) {
            for (int x = 0; x < flow.cols; x += FLOW_ARROW_STEP) {
                if (!isPointValid(cv::Point(x, y), flow.rows, flow.cols)) {
                    continue;
                }
                
                const cv::Point2f& f = flow.at<cv::Point2f>(y, x);
                cv::Point p1(x, y);
                cv::Point p2(cvRound(x + f.x * FLOW_SCALE), cvRound(y + f.y * FLOW_SCALE));
                cv::line(display, p1, p2, FLOW_ARROW_COLOR, 1);
            }
        }
    }
}

int main(int argc, char** argv) {
    // Parse arguments
    Options opt;
    if (!parseArguments(argc, argv, opt)) {
        return EXIT_FAILURE;
    }

    if (opt.showHelp) {
        showHelp(argv[0]);
        return EXIT_SUCCESS;
    }

    // Register Ctrl-C handler
    std::signal(SIGINT, handleSigInt);

    LOG(LogLevel::INFO, "=== VIO Demo: Visual-Inertial Odometry ===");

    // Initialize output directory
    if (!initializeOutputDirectory()) {
        return EXIT_FAILURE;
    }

    // Open camera
    cv::VideoCapture capLeft(0);
    if (!capLeft.isOpened()) {
        LOG(LogLevel::ERROR, "Cannot open camera");
        return EXIT_FAILURE;
    }

    int width  = static_cast<int>(capLeft.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(capLeft.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = capLeft.get(cv::CAP_PROP_FPS);
    if (fps <= 0) {
        fps = VIDEO_FPS;
        LOG(LogLevel::WARNING, "Invalid camera FPS, using default: " << fps);
    }

    LOG(LogLevel::INFO, "Camera resolution: " << width << "x" << height);
    LOG(LogLevel::INFO, "Camera FPS: " << fps);

    // Open log files
    std::string ts = getTimestamp();
    std::ofstream csvFile, traceFile, combFile;
    if (!openLogFiles(ts, csvFile, traceFile, combFile)) {
        return EXIT_FAILURE;
    }

    LOG(LogLevel::INFO, "Log files created in: " << OUTPUT_DIR);

    // Initialize video writer
    std::string outVideoPath = OUTPUT_DIR + "/vio_trajectory_" + ts + ".avi";
    cv::VideoWriter writer(outVideoPath,
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          fps, cv::Size(width, height));
    
    if (!writer.isOpened()) {
        LOG(LogLevel::WARNING, "Cannot open video writer: " << outVideoPath);
    } else {
        LOG(LogLevel::INFO, "Video writer initialized: " << outVideoPath);
    }

    // Capture first frame
    cv::Mat prevLeft, prevGray;
    capLeft >> prevLeft;
    if (prevLeft.empty()) {
        LOG(LogLevel::ERROR, "Cannot read first frame");
        return EXIT_FAILURE;
    }
    cv::cvtColor(prevLeft, prevGray, cv::COLOR_BGR2GRAY);

    // Initialize trajectory
    cv::Mat traj = cv::Mat::zeros(TRAJ_HEIGHT, TRAJ_WIDTH, CV_8UC3);
    cv::Point2f position = START_POS;

    int frameCount = 0;
    int emptyFrameCount = 0;

    LOG(LogLevel::INFO, "Starting processing loop. Press ESC to exit.");

    while (!stopFlag.load(std::memory_order_acquire)) {
        cv::Mat leftFrame, rightFrame, leftGray, rightGray;
        capLeft >> leftFrame;
        
        if (leftFrame.empty()) {
            emptyFrameCount++;
            LOG(LogLevel::WARNING, "Empty frame (consecutive: " << emptyFrameCount << ")");
            
            if (emptyFrameCount >= MAX_EMPTY_FRAMES) {
                LOG(LogLevel::INFO, "End of video or too many empty frames. Stopping.");
                break;
            }
            continue;
        }
        emptyFrameCount = 0;

        // Simulate right camera with horizontal shift
        cv::Mat shiftMat = (cv::Mat_<double>(2, 3) << 1, 0, SHIFT_PIXELS, 0, 1, 0);
        cv::warpAffine(leftFrame, rightFrame, shiftMat, leftFrame.size());

        cv::cvtColor(leftFrame, leftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightFrame, rightGray, cv::COLOR_BGR2GRAY);

        // Compute motion field & disparity
        cv::Mat flow = computeVideoMotionField(prevGray, leftGray);
        cv::Mat disparity = computeVideoDisparity(leftGray, rightGray);

        cv::Scalar avgFlow = cv::mean(flow);
        cv::Scalar avgDepth = cv::mean(disparity);
        
        // Use depth to scale motion (more depth = more movement)
        float scale = 1.0f + static_cast<float>(avgDepth[0] / DEPTH_MAX_VALUE);

        // Update position with amplified flow
        position += cv::Point2f(
            static_cast<float>(avgFlow[0]) * FLOW_SCALE * scale,
            static_cast<float>(avgFlow[1]) * FLOW_SCALE * scale
        );

        // Clamp trajectory to canvas
        position.x = std::clamp(position.x, 0.0f, static_cast<float>(TRAJ_WIDTH - 1));
        position.y = std::clamp(position.y, 0.0f, static_cast<float>(TRAJ_HEIGHT - 1));

        // Draw trajectory point
        cv::circle(traj, position, TRAJ_CIRCLE_RADIUS, TRAJ_COLOR, -1);

        // Create display with trajectory overlay
        cv::Mat display = leftFrame.clone();
        cv::Mat trajSmall;
        cv::resize(traj, trajSmall, cv::Size(width / TRAJ_DISPLAY_SCALE, height / TRAJ_DISPLAY_SCALE));
        
        int overlayX = display.cols - trajSmall.cols - OVERLAY_MARGIN;
        int overlayY = OVERLAY_MARGIN;
        if (overlayX >= 0 && overlayY >= 0 && 
            overlayX + trajSmall.cols <= display.cols &&
            overlayY + trajSmall.rows <= display.rows) {
            trajSmall.copyTo(display(cv::Rect(overlayX, overlayY, trajSmall.cols, trajSmall.rows)));
        }

        // Draw flow arrows on display
        drawFlowArrows(display, flow);

        // Display windows
        cv::imshow("VIO Live Feed", display);
        cv::imshow("Trajectory", traj);
        
        if (writer.isOpened()) {
            writer.write(display);
        }

        // Get timestamp
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // Log to CSV
        csvFile << frameCount << "," << ms << "," 
                << position.x << "," << position.y << ","
                << avgFlow[0] << "," << avgFlow[1] << "," 
                << avgDepth[0] << "\n";

        // Log trace (periodically to avoid flooding)
        if (frameCount % LOG_INTERVAL == 0) {
            std::ostringstream trace;
            trace << "[TRACE] Frame " << frameCount
                  << " | pos=(" << static_cast<int>(position.x) << "," 
                  << static_cast<int>(position.y) << ")"
                  << " | avgFlow=(" << avgFlow[0] << "," << avgFlow[1] << ")"
                  << " | avgDepth=" << avgDepth[0];

            LOG(LogLevel::INFO, trace.str());
            traceFile << trace.str() << std::endl;
            combFile << trace.str() << "\n";
        }

        combFile << "CSV," << frameCount << "," << ms << "," 
                 << position.x << "," << position.y << ","
                 << avgFlow[0] << "," << avgFlow[1] << "," 
                 << avgDepth[0] << "\n";

        // Handle keyboard input
        int key = cv::waitKey(1);
        if (key == KEY_EXIT_ESC || key == KEY_EXIT_CTRL_C) {
            LOG(LogLevel::INFO, "Exit key pressed. Stopping...");
            break;
        }

        // Update previous frame (swap for efficiency)
        std::swap(prevGray, leftGray);
        frameCount++;
    }

    // Cleanup
    capLeft.release();
    writer.release();
    cv::destroyAllWindows();
    csvFile.close();
    traceFile.close();
    combFile.close();

    LOG(LogLevel::INFO, "Processing complete. Total frames: " << frameCount);
    LOG(LogLevel::INFO, "Outputs saved to: " << OUTPUT_DIR);

    return EXIT_SUCCESS;
}