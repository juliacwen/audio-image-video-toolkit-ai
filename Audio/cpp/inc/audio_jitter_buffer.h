/*
 * @file audio_jitter_buffer.h
 * @brief RTP Jitter Buffer for network audio streaming
 * @author Julia Wen (wendigilane@gmail.com)
 * @par Revision History
 * - 12-23-2025 — Initial Checkin
 * - 12-24-2025 — Add more jitter buffer config constants
 */

#ifndef AUDIO_JITTER_BUFFER_H
#define AUDIO_JITTER_BUFFER_H

#include <deque>
#include <mutex>
#include <cstdint>
#include <vector>
#include <chrono>
#include "network_rtp.h"

// ------------------ Jitter Buffer Configuration ------------------
constexpr int JITTER_BUFFER_MIN_MS = 40;      // Minimum latency (ms)
constexpr int JITTER_BUFFER_MAX_MS = 200;     // Maximum latency (ms)
constexpr int JITTER_BUFFER_TARGET_MS = 80;   // Target latency (ms)

// Jitter buffer adaptation thresholds
constexpr int JITTER_UNDERRUN_THRESHOLD = 20;      // Underruns before increasing buffer
constexpr int JITTER_OVERFLOW_THRESHOLD = 10;      // Overflows before decreasing buffer
constexpr int JITTER_INCREASE_STEP_MS = 20;        // Increase buffer by this amount
constexpr int JITTER_DECREASE_STEP_MS = 10;        // Decrease buffer by this amount
constexpr int JITTER_UNDERRUN_PRINT_INTERVAL = 10; // Print underrun message every N underruns

// Time conversion
constexpr int MS_PER_SECOND = 1000;


class AudioJitterBuffer {
public:
    AudioJitterBuffer(int sampleRate, int channels, int targetMs = 80);
    
    // Add packet to jitter buffer
    bool addPacket(const RTPPacket& packet);
    
    // Get samples from jitter buffer
    size_t getSamples(float* output, size_t requestedSamples);
    
    // Get current buffered samples
    size_t getBufferedSamples() const;
    
    // Reset buffer
    void reset();
    
    // Statistics
    struct Statistics {
        uint64_t packetsReceived = 0;  // Changed from uint32_t to prevent overflow
        uint32_t packetsLost = 0;
        uint32_t duplicates = 0;
        uint32_t underruns = 0;
        uint32_t overflow = 0;
    };
    
    Statistics getStats() const;
    int getCurrentLatencyMs() const;

private:
    void insertSilence();
    void adaptBufferSize();
    
    mutable std::mutex mutex_;
    std::deque<RTPPacket> packets_;
    
    int sampleRate_;
    int channels_;
    int targetMs_;
    size_t targetSamples_;
    size_t minSamples_;
    size_t maxSamples_;
    
    bool playing_;
    uint16_t lastSeq_;
    bool firstPacket_;
    
    Statistics stats_;
    uint32_t lastUnderruns_ = 0;
    uint32_t lastOverflow_ = 0;
};

#endif // AUDIO_JITTER_BUFFER_H