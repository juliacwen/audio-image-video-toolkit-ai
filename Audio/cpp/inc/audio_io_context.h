/**
 * @file audio_io_context.h
 * @brief Audio I/O context for real-time audio processing with network streaming
 * @author Julia Wen (wendigilane@gmail.com)
 *
 * This file defines the AudioIOContext structure which manages:
 * - Audio input/output buffers (lock-free SPSC)
 * - RNNoise denoising state per channel
 * - Voice Activity Detection (VAD) state
 * - Network jitter buffer for received audio
 * - WAV file recording (optional)
 * - Test tone generation
 * 
 * The context is shared between the PortAudio callback thread, the processing
 * thread, and the network receive thread.
 * 
 * @par Revision History
 * - 01-02-2026 — Initial Checkin
 */

#pragma once

#ifndef AUDIO_IO_CONTEXT_H
#define AUDIO_IO_CONTEXT_H

#include <vector>
#include <memory>
#include <fstream>
#include <mutex>
#include <atomic>
#include "SPSCFloatBuffer.h"
#include "audio_jitter_buffer.h"
#include "denoise_config.h"

// Forward declarations
typedef struct DenoiseState DenoiseState;
class WavWriter;
class NetworkRTP;

// ------------------ Audio IO Context ------------------
struct AudioIOContext {
    // Audio processing
    std::vector<DenoiseState*> states;
    SPSCFloatBuffer inputBuffer;
    SPSCFloatBuffer outputBuffer;
    
    // Jitter buffer for network receive
    std::unique_ptr<AudioJitterBuffer> jitterBuffer;
    
#if ENABLE_WAV_WRITING
    std::unique_ptr<WavWriter> wavInput;
    std::unique_ptr<WavWriter> wavOutput;
    std::mutex wavMutex;  // Protect concurrent WAV writes
#endif

#if ENABLE_FILE_LOGGING
    std::ofstream logFile;
#endif
    
    // Configuration
    bool bypassDenoise;
    bool enableVAD;
    bool lowPowerMode;
    bool enableWavWrite;
    float denormalGuard;
    int numChannels;
    
    // VAD state
    int vadHangoverCounter;
    bool isVoiceActive;
    float smoothedRMS;
    
    // Network state
    bool networkSend = false;
    bool networkReceive = false;
    std::unique_ptr<NetworkRTP> networkRTP;
    
    // Test tone generator
    bool generateTestTone = false;
    double testTonePhase = 0.0;

    AudioIOContext(size_t bufferSize, int numCh, bool bypass, bool vad, bool lowPower, bool wavWrite);
    ~AudioIOContext();
    
    bool initWavWriters(bool hasInput, bool hasOutput, bool isSender, bool isReceiver);
};

#endif // AUDIO_IO_CONTEXT_H