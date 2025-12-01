# Makefile for Video Processing Tools
# Compatible with macOS Homebrew OpenCV installation

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra

# Use manual OpenCV installation (source tree structure)
OPENCV_PREFIX = /usr/local/opencv

# Include paths for source tree installation (headers in modules/*/include/)
OPENCV_INCLUDE = -I$(OPENCV_PREFIX)/include \
                 -I$(OPENCV_PREFIX)/build \
                 -I$(OPENCV_PREFIX)/modules/core/include \
                 -I$(OPENCV_PREFIX)/modules/calib3d/include \
                 -I$(OPENCV_PREFIX)/modules/features2d/include \
                 -I$(OPENCV_PREFIX)/modules/flann/include \
                 -I$(OPENCV_PREFIX)/modules/dnn/include \
                 -I$(OPENCV_PREFIX)/modules/highgui/include \
                 -I$(OPENCV_PREFIX)/modules/imgcodecs/include \
                 -I$(OPENCV_PREFIX)/modules/videoio/include \
                 -I$(OPENCV_PREFIX)/modules/imgproc/include \
                 -I$(OPENCV_PREFIX)/modules/ml/include \
                 -I$(OPENCV_PREFIX)/modules/objdetect/include \
                 -I$(OPENCV_PREFIX)/modules/photo/include \
                 -I$(OPENCV_PREFIX)/modules/stitching/include \
                 -I$(OPENCV_PREFIX)/modules/video/include

OPENCV_LIBS = -L$(OPENCV_PREFIX)/build/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_calib3d -lopencv_imgcodecs
OPENCV_LIB_DIR = $(OPENCV_PREFIX)/build/lib

# Add rpath for runtime library loading on macOS
LDFLAGS = $(OPENCV_LIBS) -Wl,-rpath,$(OPENCV_LIB_DIR)

# Update CXXFLAGS with OpenCV includes and local includes
CXXFLAGS += $(OPENCV_INCLUDE) -Isrc/video_common/inc

# Directories
COMMON_INC = src/video_common/inc
COMMON_SRC = src/video_common/src

VIDEO_ENCODING_SRC = src/video_encoding/video_motion_estimation.cpp \
                     src/video_encoding/video_frame_prediction.cpp \
                     src/video_encoding/video_residual.cpp \
                     src/video_encoding/video_block_matching.cpp

VIDEO_MOTION_SRC = src/video_motion_and_depth/video_trajectory_from_motion.cpp \
                   src/video_motion_and_depth/video_depth_from_stereo.cpp \
                   src/video_motion_and_depth/video_vio_demo.cpp

# Extract base names for targets
VIDEO_ENCODING_TARGETS = $(notdir $(VIDEO_ENCODING_SRC:.cpp=))
VIDEO_MOTION_TARGETS = $(notdir $(VIDEO_MOTION_SRC:.cpp=))

# Targets
all: $(VIDEO_ENCODING_TARGETS) $(VIDEO_MOTION_TARGETS)

# Generic build rule for video encoding demos
$(VIDEO_ENCODING_TARGETS): %: src/video_encoding/%.cpp $(COMMON_SRC)/video_common.cpp
	@echo "Building $@ from $<"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Generic build rule for video motion/depth demos  
$(VIDEO_MOTION_TARGETS): %: src/video_motion_and_depth/%.cpp $(COMMON_SRC)/video_common.cpp
	@echo "Building $@ from $<"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Debug target to show detected paths
debug:
	@echo "OpenCV Prefix: $(OPENCV_PREFIX)"
	@echo "OpenCV Include: $(OPENCV_INCLUDE)"
	@echo "OpenCV Libs: $(OPENCV_LIBS)"
	@echo "OpenCV Lib Dir: $(OPENCV_LIB_DIR)"
	@echo "Encoding Targets: $(VIDEO_ENCODING_TARGETS)"
	@echo "Motion Targets: $(VIDEO_MOTION_TARGETS)"

# Clean
clean:
	rm -f $(VIDEO_ENCODING_TARGETS) $(VIDEO_MOTION_TARGETS)
	rm -rf *.dSYM

# Phony targets
.PHONY: all clean debug