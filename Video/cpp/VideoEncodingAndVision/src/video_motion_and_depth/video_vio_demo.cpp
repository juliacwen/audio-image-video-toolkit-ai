/******************************************************************************
 * File: video_vio_demo.cpp
 * Description: Simplified VIO demo combining motion vectors and depth (disparity)
 *              to visualize camera trajectory in 2D. Supports a single camera
 *              by simulating the right frame with a horizontal shift.
 *              Logs per-frame metrics (position, flow, depth) to CSV, trace,
 *              and combined log files. Verbose output via -v or --verbose.
 * Return: 0  -> success
 *         -1 -> error (cannot open camera or log files)
 * Author: [Your Name]
 * Date: 2025-09-04
 ******************************************************************************/

#include <opencv2/opencv.hpp>
#include "../../video_common/inc/video_common.h"
#include <iostream>
#include <fstream>
#include <deque>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <string>

// =================== Constants ===================
const int TRAJ_WIDTH         = 800;
const int TRAJ_HEIGHT        = 600;
const cv::Point2f START_POS(TRAJ_WIDTH/2.f, TRAJ_HEIGHT/2.f);
const int TRAJ_CIRCLE_RADIUS = 2;
const cv::Scalar TRAJ_COLOR(0, 255, 0);
const int ESC_KEY            = 27;
const int SHIFT_PIXELS       = 15;       // stronger simulated right camera shift
const int TRAJ_DISPLAY_SCALE = 4;        // Downscale trajectory overlay
const int FPS_FALLBACK       = 20;
const float FLOW_SCALE       = 20.f;     // Amplify flow for visible trajectory
const float DEPTH_MAX_VALUE  = 255.0f;   // Maximum depth value for scaling
// ==================================================

struct Options {
    bool verbose = false;
};

Options parseArgs(int argc, char** argv) {
    Options opt;
    for (int i=1; i<argc; i++) {
        std::string arg = argv[i];
        if (arg=="-v" || arg=="--verbose") opt.verbose = true;
    }
    return opt;
}

int main(int argc, char** argv) {
    Options opt = parseArgs(argc, argv);

    cv::VideoCapture capLeft(0);
    if (!capLeft.isOpened()) {
        std::cerr << "Error: Cannot open camera." << std::endl;
        return -1;
    }

    // Prepare output directories and log files
    std::filesystem::create_directories("vio_output");
    std::string ts = getTimestamp();
    std::string csvPath  = "vio_output/motion_log_" + ts + ".csv";
    std::string txtPath  = "vio_output/trace_log_" + ts + ".txt";
    std::string combPath = "vio_output/combined_log_" + ts + ".log";
    std::string outVideoPath = "vio_output/vio_trajectory_" + ts + ".avi";

    std::ofstream csvFile(csvPath);
    std::ofstream traceFile(txtPath);
    std::ofstream combFile(combPath);
    if (!csvFile.is_open() || !traceFile.is_open() || !combFile.is_open()) {
        std::cerr << "Error: cannot open log files" << std::endl;
        return -1;
    }
    csvFile << "frame,timestamp_ms,pos_x,pos_y,avg_flow_x,avg_flow_y,avg_depth\n";
    combFile << "CSV_HEADER: frame,timestamp_ms,pos_x,pos_y,avg_flow_x,avg_flow_y,avg_depth\n";

    int width  = static_cast<int>(capLeft.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(capLeft.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = capLeft.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = FPS_FALLBACK;

    cv::VideoWriter writer(outVideoPath, cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(width,height));

    cv::Mat prevLeft, prevGray;
    capLeft >> prevLeft;
    if (prevLeft.empty()) return -1;
    cv::cvtColor(prevLeft, prevGray, cv::COLOR_BGR2GRAY);

    cv::Mat traj = cv::Mat::zeros(TRAJ_HEIGHT, TRAJ_WIDTH, CV_8UC3);
    cv::Point2f position = START_POS;

    int frameCount = 0;
    while (true) {
        cv::Mat leftFrame, rightFrame, leftGray, rightGray;
        capLeft >> leftFrame;
        if (leftFrame.empty()) break;

        // Simulate right camera
        cv::Mat shiftMat = (cv::Mat_<double>(2,3) << 1,0,SHIFT_PIXELS, 0,1,0);
        cv::warpAffine(leftFrame, rightFrame, shiftMat, leftFrame.size());

        cv::cvtColor(leftFrame, leftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightFrame, rightGray, cv::COLOR_BGR2GRAY);

        // Compute motion field & disparity
        cv::Mat flow = computeVideoMotionField(prevGray, leftGray);
        cv::Mat disparity = computeVideoDisparity(leftGray, rightGray);

        cv::Scalar avgFlow = cv::mean(flow);
        cv::Scalar avgDepth = cv::mean(disparity);
        float scale = 1.0f + static_cast<float>(avgDepth[0] / DEPTH_MAX_VALUE);

        // Update position with amplified flow
        position += cv::Point2f(static_cast<float>(avgFlow[0])*FLOW_SCALE*scale,
                                static_cast<float>(avgFlow[1])*FLOW_SCALE*scale);

        // Clamp trajectory
        position.x = std::clamp(position.x, 0.f, static_cast<float>(TRAJ_WIDTH-1));
        position.y = std::clamp(position.y, 0.f, static_cast<float>(TRAJ_HEIGHT-1));

        // Draw trajectory
        cv::circle(traj, position, TRAJ_CIRCLE_RADIUS, TRAJ_COLOR, -1);

        // Window 1: Live camera feed with trajectory overlay
        cv::Mat display = leftFrame.clone();
        cv::Mat trajSmall;
        cv::resize(traj, trajSmall, cv::Size(width/TRAJ_DISPLAY_SCALE, height/TRAJ_DISPLAY_SCALE));
        trajSmall.copyTo(display(cv::Rect(display.cols-trajSmall.cols-10, 10, trajSmall.cols, trajSmall.rows)));

        // Draw flow arrows on display
        for(int y=0; y<flow.rows; y+=10){
            for(int x=0; x<flow.cols; x+=10){
                const cv::Point2f f = flow.at<cv::Point2f>(y,x);
                cv::line(display, cv::Point(x,y), cv::Point(cvRound(x+f.x*FLOW_SCALE), cvRound(y+f.y*FLOW_SCALE)), cv::Scalar(0,0,255));
            }
        }

        cv::imshow("VIO Live Feed", display);

        // Window 2: Full trajectory visualization
        cv::imshow("Trajectory", traj);

        if (writer.isOpened()) writer.write(display);

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // Logging
        csvFile << frameCount << "," << ms << "," << position.x << "," << position.y
                << "," << avgFlow[0] << "," << avgFlow[1] << "," << avgDepth[0] << "\n";

        std::ostringstream trace;
        trace << "[TRACE] Frame " << frameCount
              << " | pos=(" << position.x << "," << position.y << ")"
              << " | avgFlow=(" << avgFlow[0] << "," << avgFlow[1] << ")"
              << " | avgDepth=" << avgDepth[0];

        if (opt.verbose) std::cout << trace.str() << std::endl;
        traceFile << trace.str() << std::endl;
        combFile << trace.str() << "\n";
        combFile << "CSV," << frameCount << "," << ms << "," << position.x << "," << position.y
                 << "," << avgFlow[0] << "," << avgFlow[1] << "," << avgDepth[0] << "\n";

        int key = cv::waitKey(1);
        if (key == ESC_KEY) break;

        prevGray = leftGray.clone();
        frameCount++;
    }

    capLeft.release();
    writer.release();
    cv::destroyAllWindows();
    csvFile.close();
    traceFile.close();
    combFile.close();

    return 0;
}
