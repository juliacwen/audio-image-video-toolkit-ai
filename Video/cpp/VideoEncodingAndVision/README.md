# VideoEncodingAndVision Tools Collection

This repository contains a set of video processing demos and utilities. Each tool focuses on a specific aspect of computer vision video processing, including motion estimation, optical flow, frame prediction, stereo disparity, residual computation, and visual-inertial odometry.

All tools use OpenCV and provide consistent logging via the `-v` option (verbosity levels 1-3). Timestamp, FPS, and frame count overlays are included in outputs.

---

## Directory Structure

```
video_common/
    inc/
        video_common.h
    src/
        video_common.cpp
video_encoding/
    video_motion_estimation.cpp
    video_frame_prediction.cpp
    video_block_matching.cpp
    video_residual.cpp
video_motion_and_depth/
    video_vio_demo.cpp
    video_trajectory_from_motion.cpp
    video_depth_from_stereo.cpp
```

- **video_common/**: Shared utilities and common functions.
- **video_encoding/**: Tools for motion estimation, frame prediction, residual computation, and block matching.
- **video_motion_and_depth/**: Tools for VIO, trajectory, and stereo depth demonstrations.

---

## Tools Overview

### video_common
**Directory:** `video_common/`

- Provides shared functionality for all video processing demos.
- `video_common.h` and `video_common.cpp` contain utility functions for logging, frame overlays, and video I/O.

---

### video_motion_estimation
**Directory:** `video_encoding/`

- Captures video from a camera and computes dense optical flow using Farneback method.
- Overlays motion vectors and reference points on frames.
- Writes video to `output_video.avi`.
- Displays two windows: `Original Frame` and `Motion Vectors`.
- Logging: `-v 1` INFO, `-v 2` WARNING, `-v 3` ERROR.
- Exits on `ESC` or `Ctrl-C`.

---

### video_frame_prediction
**Directory:** `video_encoding/`

- Performs motion-compensated frame prediction using dense optical flow.
- Computes predicted frame, overlays motion vectors, and calculates MAD.
- Displays predicted frame in a window.
- Writes video to `predicted_output.avi`.
- Logging levels and graceful exit with `ESC` or `Ctrl-C`.

---

### video_block_matching
**Directory:** `video_encoding/`

- Demonstrates stereo disparity with a single camera; right frame is simulated.
- Uses OpenCV `StereoBM` for block matching.
- Displays two windows: `Original (Left)` and `Disparity Map`.
- Writes video to `video_block_output.avi`.
- Overlays timestamp, FPS, and frame number.
- Logging and exit on `ESC` or `Ctrl-C`.

---

### video_residual
**Directory:** `video_encoding/`

- Computes residual between predicted and current frames using motion estimation.
- Displays two windows: `Predicted Frame (overlay)` and `Residual Frame`.
- Writes videos to `predicted_output.avi` and `residual_output.avi`.
- Logging with `-v` option.
- Includes timestamp, FPS, frame index overlays.
- Exits on `ESC` or `Ctrl-C`.

---

### video_vio_demo
**Directory:** `video_motion_and_depth/`

- Visualizes simplified Visual-Inertial Odometry (VIO) trajectory using a single camera.
- Computes motion field and optionally a simulated depth map.
- Displays two windows: `Camera Frame` and `Trajectory`.
- Supports `-v` logging and overlays: timestamp, FPS, and position.
- Exits gracefully on `ESC` or `Ctrl-C`.

---

### video_trajectory_from_motion
**Directory:** `video_motion_and_depth/`

- Computes camera trajectory from motion estimation.
- Displays trajectory and camera motion overlay.
- Logging and video overlay included.

---

### video_depth_from_stereo
**Directory:** `video_motion_and_depth/`

- Computes depth map from stereo simulation or stereo input.
- Displays depth map alongside original frame.
- Supports logging and overlays timestamp and FPS.

---

## Logging Levels (`-v` option)

| Level | Description |
|-------|-------------|
| 1     | INFO, WARNING, ERROR |
| 2     | WARNING, ERROR |
| 3     | ERROR only |

All tools support this unified logging convention for easier debugging.

---

## Common Features Across Tools

- **Two windows** where applicable: one for raw frame and one for processed visualization.
- **Timestamp overlay** on frames for accurate temporal reference.
- **FPS computation** for performance monitoring.
- **Graceful exit** using `ESC` or `Ctrl-C`.
- **No magic numbers**: constants are named and defined at the top of each source file.
- **Video saving** in MJPG/AVI format.
- **Shared utilities** in `video_common/` for logging and overlays.

---

## Building

Use C++17 and OpenCV. Include `video_common/inc` in your include paths.

Example:
```bash
g++ -std=c++17 -I/usr/local/opencv/include -Ivideo_common/inc \
video_encoding/src/video_motion_estimation.cpp \
video_common/src/video_common.cpp -o video_motion_estimation \
-L/usr/local/opencv/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_videoio
```

---

## Usage

```bash
./video_vio_demo -v 1
./video_trajectory_from_motion -v 1
./video_depth_from_stereo -v 1
./video_motion_estimation -v 2
./video_frame_prediction -v 1
./video_block_matching -v 1
./video_residual -v 1
```

- `-v` sets the logging level.
- Windows pop up showing processed frames.
- Videos are saved in the same directory.
- `ESC` or `Ctrl-C` stops the processing.

---

## Notes

- All video capture assumes a single camera device (device 0).  
- For tools requiring two frames (e.g., stereo disparity or block matching), the second frame is **simulated** by horizontally shifting the captured frame or using previously captured frames.  
- Disparity maps in `video_block_matching` are simulated by horizontal shifts.  
- MAD (Mean Absolute Difference) and motion vector statistics are logged where applicable.  
- Constants (e.g., vector scales, thresholds) are defined at the top of each source file.  
- Designed for quick testing, educational, and demonstration purposes.  
- **Some source code was assisted with AI tools, but core logic was implemented and verified manually.**

---

## License

MIT License