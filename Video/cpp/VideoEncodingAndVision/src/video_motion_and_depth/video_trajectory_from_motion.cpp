/******************************************************************************
 * File: video_trajectory_from_motion.cpp
 * Description: Tracks average motion in video frames and visualizes trajectory.
 *              Supports live display, video recording, snapshots, and detailed logging.
 *              Smooths flow over a sliding window, clips extreme motion vectors.
 *              Fully modernized for C++17: file-local constants, chrono, filesystem, and smart usage.
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
#include <chrono>
#include <deque>
#include <string>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <csignal>
#include <algorithm>
#include <cstdlib>

using namespace video_common;

namespace {
    // =================== Constants (file-local) ===================
    constexpr int DEFAULT_TRAJ_WIDTH        = 800;
    constexpr int DEFAULT_TRAJ_HEIGHT       = 600;
    constexpr float DEFAULT_MOTION_THRESH   = 0.01f;
    constexpr float DEFAULT_MAX_FLOW_THRESH = 10.0f;
    constexpr int LINE_THICKNESS            = 2;
    constexpr int CIRCLE_RADIUS             = 3;
    constexpr int SNAPSHOT_INTERVAL         = 30; // frames
    constexpr int TRAJ_DISPLAY_SCALE        = 4;  // downscale trajectory overlay
    constexpr char SNAPSHOT_KEY             = 's';
    
    // Validation ranges
    constexpr int MIN_SMOOTH_SIZE = 1;
    constexpr int MAX_SMOOTH_SIZE = 50;
    constexpr float MIN_POS_SCALE = 1.0f;
    constexpr float MAX_POS_SCALE = 1000.0f;
    // =============================================================

    // Command-line options
    struct Options {
        std::string input = "0";
        bool saveVideo = true;
        int smoothSize = 5;
        float posScale = 100.0f;
        std::string outDir = "run_output";
        std::string calibFile = "";
        bool showHelp = false;
    };

    // Helper: generate timestamp string for unique filenames
    std::string generateTimestampFilename() {
        using namespace std::chrono;
        auto t = system_clock::now();
        auto ms = duration_cast<milliseconds>(t.time_since_epoch()) % 1000;
        std::time_t tt = system_clock::to_time_t(t);
        std::tm tm = *std::localtime(&tt);
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
        char out[96];
        std::snprintf(out, sizeof(out), "%s_%03lld", buf, static_cast<long long>(ms.count()));
        return std::string(out);
    }

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
                        ++i;
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
            } else if (arg == "--no-video") {
                opt.saveVideo = false;
            } else if (arg == "--smooth") {
                if (i + 1 >= argc) {
                    LOG(LogLevel::ERROR, "Missing value for --smooth");
                    return false;
                }
                try {
                    opt.smoothSize = std::stoi(argv[i + 1]);
                    if (opt.smoothSize < MIN_SMOOTH_SIZE || opt.smoothSize > MAX_SMOOTH_SIZE) {
                        LOG(LogLevel::ERROR, "Smooth size must be between " 
                            << MIN_SMOOTH_SIZE << " and " << MAX_SMOOTH_SIZE);
                        return false;
                    }
                    ++i;
                } catch (const std::exception& e) {
                    LOG(LogLevel::ERROR, "Invalid smooth size: " << argv[i + 1]);
                    return false;
                }
            } else if (arg == "--scale") {
                if (i + 1 >= argc) {
                    LOG(LogLevel::ERROR, "Missing value for --scale");
                    return false;
                }
                try {
                    opt.posScale = std::stof(argv[i + 1]);
                    if (opt.posScale < MIN_POS_SCALE || opt.posScale > MAX_POS_SCALE) {
                        LOG(LogLevel::ERROR, "Position scale must be between " 
                            << MIN_POS_SCALE << " and " << MAX_POS_SCALE);
                        return false;
                    }
                    ++i;
                } catch (const std::exception& e) {
                    LOG(LogLevel::ERROR, "Invalid scale: " << argv[i + 1]);
                    return false;
                }
            } else if (arg == "--out") {
                if (i + 1 >= argc) {
                    LOG(LogLevel::ERROR, "Missing value for --out");
                    return false;
                }
                opt.outDir = argv[++i];
            } else if (arg == "--calib") {
                if (i + 1 >= argc) {
                    LOG(LogLevel::ERROR, "Missing value for --calib");
                    return false;
                }
                opt.calibFile = argv[++i];
            } else if (arg == "-h" || arg == "--help") {
                opt.showHelp = true;
                return true;
            } else if (arg[0] != '-') {
                opt.input = arg;
            } else {
                LOG(LogLevel::WARNING, "Unknown argument: " << arg);
            }
        }
        return true;
    }

    void showHelp(const char* programName) {
        std::cout << "Usage: " << programName << " [options] [input]\n"
                  << "Options:\n"
                  << "  -v, --verbose <level>     Set log level (1=INFO, 2=WARNING, 3=ERROR)\n"
                  << "  --no-video                Disable video output\n"
                  << "  --smooth <size>           Smoothing window size (" 
                  << MIN_SMOOTH_SIZE << "-" << MAX_SMOOTH_SIZE << ", default=5)\n"
                  << "  --scale <factor>          Position scale factor ("
                  << MIN_POS_SCALE << "-" << MAX_POS_SCALE << ", default=100.0)\n"
                  << "  --out <directory>         Output directory (default=run_output)\n"
                  << "  --calib <file>            Camera calibration file (.yml)\n"
                  << "  -h, --help                Show this help message\n"
                  << "  [input]                   Video file or camera index (default=0)\n"
                  << "\nControls:\n"
                  << "  ESC - Exit program\n"
                  << "  's' - Save snapshot\n"
                  << "\nOutputs:\n"
                  << "  motion_log_*.csv          - Frame-by-frame motion data\n"
                  << "  trace_log_*.txt           - Detailed trace log\n"
                  << "  combined_log_*.log        - Combined CSV and trace\n"
                  << "  output_with_flow_*.avi    - Video with trajectory overlay\n"
                  << "  trajectory_*.png          - Final trajectory image\n";
    }

    bool initializeOutputDirectory(const std::string& outDir) {
        try {
            std::filesystem::create_directories(outDir);
            return true;
        } catch (const std::filesystem::filesystem_error& e) {
            LOG(LogLevel::ERROR, "Failed to create output directory: " << e.what());
            return false;
        }
    }

    bool openLogFiles(const std::string& csvPath, const std::string& txtPath,
                     const std::string& combPath, std::ofstream& csvFile,
                     std::ofstream& traceFile, std::ofstream& combFile) {
        csvFile.open(csvPath);
        traceFile.open(txtPath);
        combFile.open(combPath);
        
        if (!csvFile.is_open() || !traceFile.is_open() || !combFile.is_open()) {
            LOG(LogLevel::ERROR, "Cannot open log files");
            return false;
        }
        
        csvFile << "frame,timestamp,avg_dx,avg_dy,smoothed_dx,smoothed_dy,pos_x,pos_y,mean_magnitude\n";
        combFile << "CSV_HEADER: frame,timestamp,avg_dx,avg_dy,smoothed_dx,smoothed_dy,pos_x,pos_y,mean_magnitude\n";
        
        return true;
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

    LOG(LogLevel::INFO, "=== Video Trajectory from Motion ===");

    // Create output directory
    if (!initializeOutputDirectory(opt.outDir)) {
        return EXIT_FAILURE;
    }

    // Generate output file paths
    std::string ts = generateTimestampFilename();
    std::string csvPath       = opt.outDir + "/motion_log_"   + ts + ".csv";
    std::string txtPath       = opt.outDir + "/trace_log_"    + ts + ".txt";
    std::string combPath      = opt.outDir + "/combined_log_" + ts + ".log";
    std::string outVideoPath  = opt.outDir + "/output_with_flow_" + ts + ".avi";
    std::string trajImagePath = opt.outDir + "/trajectory_"   + ts + ".png";

    // Open log files
    std::ofstream csvFile, traceFile, combFile;
    if (!openLogFiles(csvPath, txtPath, combPath, csvFile, traceFile, combFile)) {
        return EXIT_FAILURE;
    }

    LOG(LogLevel::INFO, "Log files created in: " << opt.outDir);

    // Open video capture
    cv::VideoCapture cap;
    if (opt.input == "0") {
        cap.open(0);
    } else {
        cap.open(opt.input);
    }
    
    if (!cap.isOpened()) {
        LOG(LogLevel::ERROR, "Cannot open input: " << opt.input);
        return EXIT_FAILURE;
    }

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) {
        fps = VIDEO_FPS;
        LOG(LogLevel::WARNING, "Invalid camera FPS, using default: " << fps);
    }

    LOG(LogLevel::INFO, "Input resolution: " << width << "x" << height);
    LOG(LogLevel::INFO, "Input FPS: " << fps);

    // Load camera calibration if provided
    cv::Mat K, distCoeffs;
    if (!opt.calibFile.empty()) {
        cv::FileStorage fs(opt.calibFile, cv::FileStorage::READ);
        if (fs.isOpened()) {
            fs["camera_matrix"] >> K;
            fs["dist_coeffs"] >> distCoeffs;
            LOG(LogLevel::INFO, "Loaded calibration: " << opt.calibFile);
        } else {
            LOG(LogLevel::WARNING, "Could not open calibration file: " << opt.calibFile);
        }
    }

    // Initialize video writer
    cv::VideoWriter writer;
    if (opt.saveVideo) {
        writer.open(outVideoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 
                   fps, cv::Size(width, height));
        if (!writer.isOpened()) {
            opt.saveVideo = false;
            LOG(LogLevel::WARNING, "Cannot open video writer, video output disabled");
        } else {
            LOG(LogLevel::INFO, "Video writer initialized: " << outVideoPath);
        }
    }

    // Capture and process first frame
    cv::Mat prevFrame, prevGray;
    cap >> prevFrame;
    if (prevFrame.empty()) {
        LOG(LogLevel::ERROR, "Cannot read first frame");
        return EXIT_FAILURE;
    }
    
    if (!K.empty()) {
        cv::undistort(prevFrame, prevFrame, K, distCoeffs);
    }
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

    // Initialize trajectory image
    cv::Mat traj = cv::Mat::zeros(DEFAULT_TRAJ_HEIGHT, DEFAULT_TRAJ_WIDTH, CV_8UC3);
    cv::Point2f position(DEFAULT_TRAJ_WIDTH / 2.f, DEFAULT_TRAJ_HEIGHT / 2.f);
    cv::Point2f lastPosition = position;
    std::deque<cv::Point2f> smoothWindow;
    
    int frameCount = 0;
    int emptyFrameCount = 0;

    LOG(LogLevel::INFO, "Starting processing loop. Press ESC to exit, 's' to save snapshot.");

    while (!stopFlag.load(std::memory_order_acquire)) {
        cv::Mat currFrame, currGray;
        cap >> currFrame;
        
        if (currFrame.empty()) {
            emptyFrameCount++;
            LOG(LogLevel::WARNING, "Empty frame (consecutive: " << emptyFrameCount << ")");
            
            if (emptyFrameCount >= MAX_EMPTY_FRAMES) {
                LOG(LogLevel::INFO, "End of video or too many empty frames. Stopping.");
                break;
            }
            continue;
        }
        emptyFrameCount = 0;

        if (!K.empty()) {
            cv::undistort(currFrame, currFrame, K, distCoeffs);
        }
        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

        // Compute optical flow
        cv::Mat flow = computeVideoMotionField(prevGray, currGray);

        // Calculate average flow
        cv::Scalar avgScalar = cv::mean(flow);
        cv::Point2f avgFlow(static_cast<float>(avgScalar[0]), static_cast<float>(avgScalar[1]));

        // Extract flow components and compute magnitude
        cv::Mat flowX, flowY, mag;
        cv::extractChannel(flow, flowX, 0);
        cv::extractChannel(flow, flowY, 1);
        cv::magnitude(flowX, flowY, mag);

        double meanMag = cv::mean(mag)[0];
        double maxFlowMag;
        cv::minMaxLoc(mag, nullptr, &maxFlowMag);

        // Clip extreme flow vectors
        cv::Mat clippedX = flowX.clone(), clippedY = flowY.clone();
        int contribPts = 0;
        for (int y = 0; y < flowX.rows; ++y) {
            for (int x = 0; x < flowX.cols; ++x) {
                float fx = flowX.at<float>(y, x);
                float fy = flowY.at<float>(y, x);
                double flowMag = std::hypot(fx, fy);
                
                if (flowMag > DEFAULT_MAX_FLOW_THRESH) {
                    double scale = DEFAULT_MAX_FLOW_THRESH / flowMag;
                    clippedX.at<float>(y, x) = static_cast<float>(fx * scale);
                    clippedY.at<float>(y, x) = static_cast<float>(fy * scale);
                } else {
                    contribPts++;
                }
            }
        }

        cv::Point2f clippedAvgFlow(
            static_cast<float>(cv::mean(clippedX)[0]),
            static_cast<float>(cv::mean(clippedY)[0])
        );

        // Apply smoothing
        smoothWindow.push_back(clippedAvgFlow);
        if (smoothWindow.size() > static_cast<size_t>(opt.smoothSize)) {
            smoothWindow.pop_front();
        }
        
        cv::Point2f smoothFlow(0, 0);
        for (const auto& p : smoothWindow) {
            smoothFlow += p;
        }
        smoothFlow *= (1.0f / smoothWindow.size());
        
        // Apply motion threshold
        if (std::hypot(smoothFlow.x, smoothFlow.y) < DEFAULT_MOTION_THRESH) {
            smoothFlow = {0, 0};
        }

        // Update position
        lastPosition = position;
        position += smoothFlow * opt.posScale;
        position.x = std::clamp(position.x, 0.f, static_cast<float>(DEFAULT_TRAJ_WIDTH - 1));
        position.y = std::clamp(position.y, 0.f, static_cast<float>(DEFAULT_TRAJ_HEIGHT - 1));

        // Draw trajectory
        cv::line(traj, lastPosition, position, cv::Scalar(0, 0, 255), LINE_THICKNESS);
        cv::circle(traj, position, CIRCLE_RADIUS, cv::Scalar(0, 255, 0), -1);

        // Create display with trajectory overlay
        cv::Mat display = currFrame.clone();
        cv::Mat trajSmall;
        cv::resize(traj, trajSmall, cv::Size(width / TRAJ_DISPLAY_SCALE, height / TRAJ_DISPLAY_SCALE));
        
        int overlayX = display.cols - trajSmall.cols - 10;
        int overlayY = 10;
        if (overlayX >= 0 && overlayY >= 0 && 
            overlayX + trajSmall.cols <= display.cols &&
            overlayY + trajSmall.rows <= display.rows) {
            trajSmall.copyTo(display(cv::Rect(overlayX, overlayY, trajSmall.cols, trajSmall.rows)));
        }

        // Display windows
        cv::imshow("Live Feed", display);
        cv::imshow("Trajectory", traj);
        
        if (opt.saveVideo) {
            writer.write(display);
        }

        // Log data
        std::string ts_now = getTimestamp();
        csvFile << frameCount << "," << ts_now << "," 
                << avgFlow.x << "," << avgFlow.y << ","
                << smoothFlow.x << "," << smoothFlow.y << "," 
                << position.x << "," << position.y << "," << meanMag << "\n";

        std::ostringstream trace;
        trace << "[TRACE] Frame " << frameCount
              << " | avg=(" << avgFlow.x << "," << avgFlow.y << ")"
              << " | clippedAvg=(" << clippedAvgFlow.x << "," << clippedAvgFlow.y << ")"
              << " | smooth=(" << smoothFlow.x << "," << smoothFlow.y << ")"
              << " | pos=(" << static_cast<int>(position.x) << "," 
              << static_cast<int>(position.y) << ")"
              << " | maxFlowMag=" << maxFlowMag
              << " | contribPts=" << contribPts
              << " | meanMag=" << meanMag;

        if (frameCount % SNAPSHOT_INTERVAL == 0) {
            LOG(LogLevel::INFO, trace.str());
            traceFile << trace.str() << std::endl;
            combFile << trace.str() << std::endl;
        }

        combFile << "CSV," << frameCount << "," << ts_now << "," 
                 << avgFlow.x << "," << avgFlow.y << ","
                 << smoothFlow.x << "," << smoothFlow.y << "," 
                 << position.x << "," << position.y << "," << meanMag << "\n";

        // Handle keyboard input
        int key = cv::waitKey(1);
        if (key == KEY_EXIT_ESC || key == KEY_EXIT_CTRL_C) {
            LOG(LogLevel::INFO, "Exit key pressed. Stopping...");
            break;
        }
        if (key == SNAPSHOT_KEY) {
            std::string snapPath = opt.outDir + "/snapshot_" + ts + "_f" + 
                                  std::to_string(frameCount) + ".png";
            cv::imwrite(snapPath, display);
            std::string msg = "[TRACE] Saved snapshot: " + snapPath;
            LOG(LogLevel::INFO, msg);
            traceFile << msg << std::endl;
            combFile << msg << std::endl;
        }

        // Update previous frame (swap for efficiency)
        std::swap(prevGray, currGray);
        frameCount++;
    }

    // Cleanup
    if (opt.saveVideo) {
        writer.release();
    }
    
    cv::imwrite(trajImagePath, traj);
    LOG(LogLevel::INFO, "Saved trajectory image: " << trajImagePath);

    csvFile.close();
    traceFile.close();
    combFile.close();

    cap.release();
    cv::destroyAllWindows();

    LOG(LogLevel::INFO, "Processing complete. Total frames: " << frameCount);
    LOG(LogLevel::INFO, "Outputs saved to: " << opt.outDir);

    return EXIT_SUCCESS;
}