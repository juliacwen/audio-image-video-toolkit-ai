# Makefile for VideoEncodingAndApplications
# Author: Julia wen
# Date: 2025-08-29


CXX = g++
CXXFLAGS = -std=c++17 -I/usr/local/opencv/include -I/usr/local/opencv/build -I/usr/local/opencv/modules/core/include/ -I/usr/local/opencv/modules/calib3d/include/ -I/usr/local/opencv/modules/features2d/include/ -I/usr/local/opencv/modules/flann/include/ -I/usr/local/opencv/modules/dnn/include/ -I/usr/local/opencv/modules/highgui/include/ -I/usr/local/opencv/modules/imgcodecs/include/ -I/usr/local/opencv/modules/videoio/include -I /usr/local/opencv/modules/imgproc/include/ -I/usr/local/opencv/modules/ml/include/ -I/usr/local/opencv/modules/objdetect/include/ -I /usr/local/opencv/modules/photo/include/ -I/usr/local/opencv/modules/stitching/include -I/usr/local/opencv/modules/video/include/ -I/src/video_common/inc
LDFLAGS  = -L/usr/local/opencv/build/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_calib3d -lopencv_imgcodecs


# Directories
COMMON_INC = src/video_common/inc
COMMON_SRC = src/video_common/src

VIDEO_ENCODING_SRC = src/video_encoding/video_motion_estimation.cpp \
                     src/video_encoding/video_frame_prediction.cpp \
                     src/video_encoding/video_residual.cpp \
                     Src/video_encoding/video_block_matching.cpp

VIDEO_MOTION_SRC = src/video_motion_and_depth/video_trajectory_from_motion.cpp \
                   src/video_motion_and_depth/video_depth_from_stereo.cpp \
                   src/video_motion_and_depth/video_vio_demo.cpp

# Targets
all: $(VIDEO_ENCODING_SRC:.cpp=) $(VIDEO_MOTION_SRC:.cpp=)

# Generic build rule for video demos
%: %.cpp $(COMMON_SRC)/video_common.cpp
	$(CXX) $(CXXFLAGS) -I$(COMMON_INC) -I. $^ -o $(@F) $(LDFLAGS)  # <- output to root using $(@F)


# Clean
clean:
	rm -f $(notdir $(VIDEO_ENCODING_SRC:.cpp=)) $(notdir $(VIDEO_MOTION_SRC:.cpp=))


