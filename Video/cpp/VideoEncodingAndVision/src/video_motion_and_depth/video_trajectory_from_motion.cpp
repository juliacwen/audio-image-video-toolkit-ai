/******************************************************************************
 * File: video_trajectory_from_motion.cpp
 * Description: Tracks average motion in video frames and visualizes trajectory.
 *              Supports live display, video recording, snapshots, and detailed logging.
 *              Smooths flow over a sliding window, clips extreme motion vectors.
 *              Fully modernized for C++17: file-local constants, chrono, filesystem, and smart usage.
 * Return: 0  -> success
 *         -1 -> error (cannot open camera or log files)
 * Author: Julia Wen wendigilane@gmail.com
 * Date: 2025-09-05
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

// =================== Constants (file-local) ===================
const int DEFAULT_TRAJ_WIDTH        = 800;
const int DEFAULT_TRAJ_HEIGHT       = 600;
const double DEFAULT_FPS            = 30.0;
const float DEFAULT_MOTION_THRESH   = 0.01f;
const float DEFAULT_MAX_FLOW_THRESH = 10.0f;
const int LINE_THICKNESS            = 2;
const int CIRCLE_RADIUS             = 3;
const int SNAPSHOT_INTERVAL         = 30; // frames
const int TRAJ_DISPLAY_SCALE        = 4;  // downscale trajectory overlay
// =============================================================

// Running flag
static volatile bool g_running = true;
void signalHandler(int) { g_running = false; }

// Helper: generate timestamp string for unique filenames
static std::string now_string() {
    using namespace std::chrono;
    auto t = system_clock::now();
    auto ms = duration_cast<milliseconds>(t.time_since_epoch()) % 1000;
    std::time_t tt = system_clock::to_time_t(t);
    std::tm tm = *std::localtime(&tt);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    char out[96];
    std::snprintf(out, sizeof(out), "%s_%03lld", buf, (long long)ms.count());
    return std::string(out);
}

// Command-line options
struct Options {
    std::string input = "0";
    bool saveVideo = true;
    int smoothSize = 5;
    float posScale = 100.0f;
    std::string outDir = "run_output";
    std::string calibFile = "";
    bool verbose = false;
};

Options parseArgs(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-video") opt.saveVideo = false;
        else if (arg == "--smooth" && i+1 < argc) opt.smoothSize = std::stoi(argv[++i]);
        else if (arg == "--scale" && i+1 < argc) opt.posScale = std::stof(argv[++i]);
        else if (arg == "--out" && i+1 < argc) opt.outDir = argv[++i];
        else if (arg == "--calib" && i+1 < argc) opt.calibFile = argv[++i];
        else if (arg == "--verbose") opt.verbose = true;
        else if (arg[0] != '-') opt.input = arg;
    }
    return opt;
}

int main(int argc, char** argv) {
    std::signal(SIGINT, signalHandler);
    Options opt = parseArgs(argc, argv);
    std::filesystem::create_directories(opt.outDir);

    std::string ts = now_string();
    std::string csvPath  = opt.outDir + "/motion_log_"   + ts + ".csv";
    std::string txtPath  = opt.outDir + "/trace_log_"    + ts + ".txt";
    std::string combPath = opt.outDir + "/combined_log_" + ts + ".log";
    std::string outVideoPath = opt.outDir + "/output_with_flow_" + ts + ".avi";
    std::string trajImagePath= opt.outDir + "/trajectory_"   + ts + ".png";

    std::ofstream csvFile(csvPath);
    std::ofstream traceFile(txtPath);
    std::ofstream combFile(combPath);
    if (!csvFile.is_open() || !traceFile.is_open() || !combFile.is_open()) {
        std::cerr << "Error: cannot open log files" << std::endl;
        return -1;
    }
    csvFile  << "frame,timestamp,avg_dx,avg_dy,smoothed_dx,smoothed_dy,pos_x,pos_y,mean_magnitude\n";
    combFile << "CSV_HEADER: frame,timestamp,avg_dx,avg_dy,smoothed_dx,smoothed_dy,pos_x,pos_y,mean_magnitude\n";

    cv::VideoCapture cap;
    if (opt.input == "0") cap.open(0);
    else cap.open(opt.input);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open input: " << opt.input << std::endl;
        return -1;
    }

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = DEFAULT_FPS;

    cv::Mat K, distCoeffs;
    if (!opt.calibFile.empty()) {
        cv::FileStorage fs(opt.calibFile, cv::FileStorage::READ);
        if (fs.isOpened()) {
            fs["camera_matrix"] >> K;
            fs["dist_coeffs"] >> distCoeffs;
            std::cout << "Loaded calibration: " << opt.calibFile << std::endl;
        }
    }

    cv::VideoWriter writer;
    if (opt.saveVideo) {
        writer.open(outVideoPath, cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(width,height));
        if (!writer.isOpened()) { opt.saveVideo=false; std::cerr << "Cannot open video writer\n"; }
    }

    cv::Mat prevFrame, prevGray;
    cap >> prevFrame;
    if (prevFrame.empty()) return -1;
    if (!K.empty()) cv::undistort(prevFrame, prevFrame, K, distCoeffs);
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

    cv::Mat traj = cv::Mat::zeros(DEFAULT_TRAJ_HEIGHT, DEFAULT_TRAJ_WIDTH, CV_8UC3);
    cv::Point2f position(DEFAULT_TRAJ_WIDTH/2.f, DEFAULT_TRAJ_HEIGHT/2.f);
    cv::Point2f lastPosition = position;
    std::deque<cv::Point2f> smoothWindow;
    int frameCount=0;

    while (g_running) {
        cv::Mat currFrame, currGray;
        cap >> currFrame;
        if (currFrame.empty()) break;
        if (!K.empty()) cv::undistort(currFrame, currFrame, K, distCoeffs);
        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

        cv::Mat flow = computeVideoMotionField(prevGray, currGray);

        cv::Scalar avgScalar = cv::mean(flow);
        cv::Point2f avgFlow((float)avgScalar[0], (float)avgScalar[1]);

        cv::Mat flowX, flowY, mag;
        cv::extractChannel(flow, flowX, 0);
        cv::extractChannel(flow, flowY, 1);
        cv::magnitude(flowX, flowY, mag);

        double meanMag = cv::mean(mag)[0];
        double maxFlowMag;
        cv::minMaxLoc(mag,nullptr,&maxFlowMag);

        cv::Mat clippedX = flowX.clone(), clippedY = flowY.clone();
        int contribPts=0;
        for(int y=0;y<flowX.rows;y++){
            for(int x=0;x<flowX.cols;x++){
                float fx = flowX.at<float>(y,x);
                float fy = flowY.at<float>(y,x);
                double mag = std::hypot(fx,fy);
                if(mag>DEFAULT_MAX_FLOW_THRESH){
                    double scale=DEFAULT_MAX_FLOW_THRESH/mag;
                    clippedX.at<float>(y,x)=fx*scale;
                    clippedY.at<float>(y,x)=fy*scale;
                } else contribPts++;
            }
        }

        cv::Point2f clippedAvgFlow(
            static_cast<float>(cv::mean(clippedX)[0]),
            static_cast<float>(cv::mean(clippedY)[0])
        );

        smoothWindow.push_back(clippedAvgFlow);
        if (smoothWindow.size() > (size_t)opt.smoothSize) smoothWindow.pop_front();
        cv::Point2f smoothFlow(0,0);
        for(auto&p:smoothWindow) smoothFlow+=p;
        smoothFlow *= (1.0f/smoothWindow.size());
        if(std::hypot(smoothFlow.x,smoothFlow.y)<DEFAULT_MOTION_THRESH) smoothFlow={0,0};

        lastPosition=position;
        position += smoothFlow * opt.posScale;
        position.x=std::clamp(position.x,0.f,(float)DEFAULT_TRAJ_WIDTH-1);
        position.y=std::clamp(position.y,0.f,(float)DEFAULT_TRAJ_HEIGHT-1);
        cv::line(traj,lastPosition,position,cv::Scalar(0,0,255),LINE_THICKNESS);
        cv::circle(traj,position,CIRCLE_RADIUS,cv::Scalar(0,255,0),-1);

        cv::Mat display = currFrame.clone();
        cv::Mat trajSmall;
        cv::resize(traj, trajSmall, cv::Size(width/TRAJ_DISPLAY_SCALE,height/TRAJ_DISPLAY_SCALE));
        trajSmall.copyTo(display(cv::Rect(display.cols-trajSmall.cols-10,10,trajSmall.cols,trajSmall.rows)));

        cv::imshow("Live Feed", display);
        cv::imshow("Trajectory", traj);
        if(opt.saveVideo) writer.write(display);

        std::string ts_now = getTimestamp();

        csvFile << frameCount << "," << ts_now << "," << avgFlow.x << "," << avgFlow.y << ","
                << smoothFlow.x << "," << smoothFlow.y << "," << position.x << "," << position.y
                << "," << meanMag << "\n";

        std::ostringstream trace;
        trace << "[TRACE] Frame "<<frameCount
              <<" | avg=("<<avgFlow.x<<","<<avgFlow.y<<")"
              <<" | clippedAvg=("<<clippedAvgFlow.x<<","<<clippedAvgFlow.y<<")"
              <<" | smooth=("<<smoothFlow.x<<","<<smoothFlow.y<<")"
              <<" | pos=("<<(int)position.x<<","<<(int)position.y<<")"
              <<" | maxFlowMag="<<maxFlowMag
              <<" | contribPts="<<contribPts
              <<" | meanMag="<<meanMag;

        if(opt.verbose||frameCount%SNAPSHOT_INTERVAL==0){
            std::cout<<trace.str()<<std::endl;
            traceFile<<trace.str()<<std::endl;
            combFile<<trace.str()<<std::endl;
        }

        combFile<<"CSV,"<<frameCount<<","<<ts_now<<","<<avgFlow.x<<","<<avgFlow.y<<","
                <<smoothFlow.x<<","<<smoothFlow.y<<","<<position.x<<","<<position.y
                <<","<<meanMag<<"\n";

        int key=cv::waitKey(1);
        if(key==27) break; // ESC to exit
        if(key=='s'){
            std::string snapPath=opt.outDir+"/snapshot_"+ts+"_f"+std::to_string(frameCount)+".png";
            cv::imwrite(snapPath,display);
            std::string msg="[TRACE] Saved snapshot: "+snapPath;
            std::cout<<msg<<std::endl;
            traceFile<<msg<<std::endl;
            combFile<<msg<<std::endl;
        }

        prevGray=currGray.clone();
        frameCount++;
    }

    if(opt.saveVideo) writer.release();
    cv::imwrite(trajImagePath,traj);
    std::cout<<"Saved trajectory image: "<<trajImagePath<<"\n";

    csvFile.close();
    traceFile.close();
    combFile.close();

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

