/*
 * @file audio_jitter_buffer.h
 * @brief RTP Jitter Buffer for network audio streaming
 * @author Julia Wen (wendigilane@gmail.com)
 * @par Revision History
 * - 12-23-2025 — Initial Checkin
 */

#ifndef AUDIO_JITTER_BUFFER_H
#define AUDIO_JITTER_BUFFER_H

#include <deque>
#include <mutex>
#include <cstdint>
#include <vector>
#include <chrono>

// RTP Header structure
#pragma pack(push, 1)
struct RTPHeader {
    uint8_t vpxcc;           // version(2), padding(1), extension(1), csrc count(4)
    uint8_t mpt;             // marker(1), payload type(7)
    uint16_t sequenceNumber; // Network byte order
    uint32_t timestamp;      // Network byte order
    uint32_t ssrc;           // Network byte order
    
    // Constructor to initialize all fields
    RTPHeader() 
        : vpxcc(0x80)           // version 2, no padding, no extension, no CSRC
        , mpt(111)              // Payload type (Opus default, using for raw audio)
        , sequenceNumber(0)
        , timestamp(0)
        , ssrc(0)
    {}
};
#pragma pack(pop)

// RTP Packet structure
struct RTPPacket {
    RTPHeader header;
    std::vector<uint8_t> payload;
    std::chrono::steady_clock::time_point arrivalTime;
};

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
        uint32_t packetsReceived = 0;
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