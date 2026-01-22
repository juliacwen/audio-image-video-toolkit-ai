/**
 * @file audio_framework_base.h
 * @brief Shared audio processing framework for real-time applications
 * @author Julia Wen (wendigilane@gmail.com)
 * 
 * Provides common infrastructure for real-time audio processing applications:
 * - PortAudio integration with lock-free buffers
 * - WAV recording support
 * - CSV/text logging
 * - Signal handling
 * - Command-line argument parsing
 * 
 * Used by:
 * - live_audio_denoise.cpp
 * - live_audio_tonal_detection.cpp
 * - Other real-time audio processing tools
 * 
 * @par Revision History
 * - 01-20-2026 — Initial creation - extracted from live_audio_denoise.cpp
 */

#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <atomic>
#include <vector>
#include <fstream>
#include <memory>
#include <cmath>
#include "../inc/denoise_config.h"
#include "../inc/SPSCFloatBuffer.h"
#include "portaudio.h"

#if ENABLE_WAV_WRITING
#include "../inc/wav_writer.h"
#endif

namespace AudioFramework {

// ------------------ Signal Handling ------------------
extern std::atomic<bool> keepRunning;
void setupSignalHandler();

// ------------------ Base Audio Context ------------------
/**
 * BaseAudioContext - Common audio I/O infrastructure
 * 
 * Provides ring buffers, WAV recording, and logging facilities
 * for real-time audio applications.
 */
class BaseAudioContext {
public:
    // Audio buffers (lock-free SPSC)
    SPSCFloatBuffer inputBuffer;
    SPSCFloatBuffer outputBuffer;
    
#if ENABLE_WAV_WRITING
    std::unique_ptr<WavWriter> wavInput;
    std::unique_ptr<WavWriter> wavOutput;
#endif

#if ENABLE_FILE_LOGGING
    std::ofstream logFile;
    std::string logFilename;
#endif
    
    // Configuration
    bool enableWavWrite;
    bool enableCsvLog;
    float denormalGuard;
    int numChannels;
    
    BaseAudioContext(size_t bufferSize, int numCh, bool wavWrite, bool csvLog);
    virtual ~BaseAudioContext() = default;
    
    // Initialize WAV writers
    bool initWavWriters(const std::filesystem::path& outputDir, 
                       const std::string& inputName, 
                       const std::string& outputName,
                       int sampleRate,
                       int bitDepth = 16);
    
    // Initialize log file (text or CSV)
    bool initLogFile(const std::filesystem::path& outputDir,
                    const std::string& logName,
                    const std::string& csvHeader,
                    const std::string& textHeader);
    
    // Convert text log to CSV on shutdown
    void convertLogToCSV(const std::filesystem::path& outputDir,
                        const std::string& logBaseName);
};

// ------------------ PortAudio Callback Helper ------------------
/**
 * Generic PortAudio callback that works with any BaseAudioContext
 */
int portAudioCallback(const void* inputBuffer, 
                     void* outputBuffer,
                     unsigned long framesPerBuffer,
                     const PaStreamCallbackTimeInfo*,
                     PaStreamCallbackFlags,
                     void* userData);

// ------------------ Utility Functions ------------------
/**
 * Fast RMS calculation with loop unrolling
 */
inline float calculateRMS(const float* data, size_t count, int stride = 1) {
    float sum = 0.0f;
    size_t i = 0;
    size_t count4 = (count / 4) * 4;
    
    // Unrolled loop: process 4 samples at once
    for (; i < count4; i += 4) {
        float s0 = data[i * stride];
        float s1 = data[(i + 1) * stride];
        float s2 = data[(i + 2) * stride];
        float s3 = data[(i + 3) * stride];
        sum += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
    }
    
    // Handle remaining samples
    for (; i < count; ++i) {
        float s = data[i * stride];
        sum += s * s;
    }
    
    return std::sqrt(sum / count);
}

/**
 * Convert multi-channel to mono (average)
 */
void convertToMono(const std::vector<float>& multiChannel, 
                  std::vector<float>& mono,
                  int numChannels,
                  int frameSize);

/**
 * Initialize PortAudio
 */
bool initPortAudio();

/**
 * Cleanup PortAudio
 */
void cleanupPortAudio();

/**
 * Open default PortAudio stream
 */
PaStream* openAudioStream(int inputChannels, 
                         int outputChannels,
                         int sampleRate,
                         int frameSize,
                         PaStreamCallback* callback,
                         void* userData);

/**
 * Common command-line argument structure
 */
struct CommonArgs {
    std::filesystem::path outputDir = "test_output";
    int numChannels = 1;
    bool enableWavWrite = false;
    bool enableCsvLog = false;
    int bitDepth = 16;
};

/**
 * Parse common command-line arguments
 * Returns false if help was requested or error occurred
 */
bool parseCommonArgs(int& argc, char**& argv, CommonArgs& args);

} // namespace AudioFramework