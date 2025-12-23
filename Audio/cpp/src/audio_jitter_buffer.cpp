/*
 * @file audio_jitter_buffer.cpp
 * @brief RTP Jitter Buffer implementation
 * @author Julia Wen (wendigilane@gmail.com)
 * @par Revision History
 * - 12-23-2025 — Initial Checkin
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include "../inc/audio_jitter_buffer.h"
#include "../inc/denoise_config.h"

AudioJitterBuffer::AudioJitterBuffer(int sampleRate, int channels, int targetMs)
    : sampleRate_(sampleRate)
    , channels_(channels)
    , targetMs_(targetMs)
    , playing_(false)
    , lastSeq_(0)
    , firstPacket_(true)
{
    targetSamples_ = (sampleRate * targetMs) / MS_PER_SECOND * channels;
    minSamples_ = (sampleRate * JITTER_BUFFER_MIN_MS) / MS_PER_SECOND * channels;
    maxSamples_ = (sampleRate * JITTER_BUFFER_MAX_MS) / MS_PER_SECOND * channels;
}

bool AudioJitterBuffer::addPacket(const RTPPacket& packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check for duplicate
    if (!firstPacket_ && packet.header.sequenceNumber <= lastSeq_) {
        stats_.duplicates++;
        return false;
    }
    
    // Detect packet loss
    if (!firstPacket_) {
        uint16_t expected = (lastSeq_ + 1) & 0xFFFF;
        if (packet.header.sequenceNumber != expected) {
            uint16_t lost = (packet.header.sequenceNumber - expected) & 0xFFFF;
            stats_.packetsLost += lost;
            
            // Insert silence for lost packets
            for (uint16_t seq = expected; seq != packet.header.sequenceNumber; seq = (seq + 1) & 0xFFFF) {
                insertSilence();
            }
        }
    }
    
    lastSeq_ = packet.header.sequenceNumber;
    firstPacket_ = false;
    
    // Add packet to buffer (already sorted by arrival time)
    packets_.push_back(packet);
    stats_.packetsReceived++;
    
    // Start playing when buffer reaches target
    if (!playing_ && getBufferedSamples() >= targetSamples_) {
        playing_ = true;
        // Only print once, not repeatedly
        static bool printed = false;
        if (!printed) {
            std::cout << "[AudioJitterBuffer] Started playback (buffered " 
                      << getBufferedSamples() << " samples, " << targetMs_ << "ms)\n";
            printed = true;
        }
    }
    
    // Prevent buffer overflow
    if (getBufferedSamples() > maxSamples_) {
        // Drop oldest packet
        if (!packets_.empty()) {
            packets_.pop_front();
            stats_.overflow++;
        }
    }
    
    return true;
}

size_t AudioJitterBuffer::getSamples(float* output, size_t requestedSamples) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!playing_) {
        // Not enough buffered, return silence
        std::fill(output, output + requestedSamples, 0.0f);
        // Don't increment underruns during initial buffering
        if (stats_.packetsReceived > 0) {
            stats_.underruns++;
        }
        return requestedSamples;
    }
    
    size_t samplesRetrieved = 0;
    
    while (samplesRetrieved < requestedSamples && !packets_.empty()) {
        RTPPacket& packet = packets_.front();
        size_t packetSamples = packet.payload.size() / sizeof(float);
        size_t samplesNeeded = requestedSamples - samplesRetrieved;
        size_t samplesToCopy = std::min(packetSamples, samplesNeeded);
        
        // Copy samples from packet
        std::memcpy(output + samplesRetrieved, 
                   packet.payload.data(), 
                   samplesToCopy * sizeof(float));
        
        samplesRetrieved += samplesToCopy;
        
        // Remove packet if fully consumed
        if (samplesToCopy >= packetSamples) {
            packets_.pop_front();
        } else {
            // Partial consume - remove consumed samples
            packet.payload.erase(packet.payload.begin(), 
                                packet.payload.begin() + samplesToCopy * sizeof(float));
        }
    }
    
    // Check for underrun
    if (samplesRetrieved < requestedSamples) {
        // Fill remaining with silence
        std::fill(output + samplesRetrieved, 
                 output + requestedSamples, 0.0f);
        stats_.underruns++;
        
        // Reset to buffering state if buffer is empty
        if (packets_.empty() && playing_) {
            playing_ = false;
            // Reduce console spam - only print every Nth underrun
            if (stats_.underruns % JITTER_UNDERRUN_PRINT_INTERVAL == 0) {
                std::cout << "[AudioJitterBuffer] Underrun #" << stats_.underruns 
                          << " - rebuffering\n";
            }
        }
    }
    
    // Adaptive buffer adjustment (less aggressive)
    adaptBufferSize();
    
    return requestedSamples;
}

size_t AudioJitterBuffer::getBufferedSamples() const {
    size_t total = 0;
    for (const auto& packet : packets_) {
        total += packet.payload.size() / sizeof(float);
    }
    return total;
}

void AudioJitterBuffer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    packets_.clear();
    playing_ = false;
    firstPacket_ = true;
    lastSeq_ = 0;
}

AudioJitterBuffer::Statistics AudioJitterBuffer::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

int AudioJitterBuffer::getCurrentLatencyMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t buffered = getBufferedSamples();
    if (buffered == 0) return 0;
    return (buffered * MS_PER_SECOND) / (sampleRate_ * channels_);
}

void AudioJitterBuffer::insertSilence() {
    // Insert one frame of silence for lost packet
    RTPPacket silencePacket;
    size_t silenceSize = FRAME_SIZE * channels_ * sizeof(float);
    silencePacket.payload.resize(silenceSize, 0);
    packets_.push_back(silencePacket);
}

void AudioJitterBuffer::adaptBufferSize() {
    // Less aggressive adaptation - only adjust every N underruns
    if (stats_.underruns > lastUnderruns_ + JITTER_UNDERRUN_THRESHOLD) {
        // Increase target if experiencing underruns
        int newTarget = std::min(targetMs_ + JITTER_INCREASE_STEP_MS, JITTER_BUFFER_MAX_MS);
        if (newTarget != targetMs_) {
            targetMs_ = newTarget;
            targetSamples_ = (sampleRate_ * targetMs_) / MS_PER_SECOND * channels_;
            lastUnderruns_ = stats_.underruns;
            std::cout << "[AudioJitterBuffer] Increased target to " << targetMs_ << "ms "
                      << "(underruns: " << stats_.underruns << ")\n";
        }
    }
    
    size_t buffered = getBufferedSamples();
    
    // Only decrease if we have consistently good performance
    if (buffered > targetSamples_ * 2 && stats_.overflow > lastOverflow_ + JITTER_OVERFLOW_THRESHOLD) {
        // Decrease target if buffer is consistently full
        int newTarget = std::max(targetMs_ - JITTER_DECREASE_STEP_MS, JITTER_BUFFER_MIN_MS);
        if (newTarget != targetMs_) {
            targetMs_ = newTarget;
            targetSamples_ = (sampleRate_ * targetMs_) / MS_PER_SECOND * channels_;
            lastOverflow_ = stats_.overflow;
            std::cout << "[AudioJitterBuffer] Decreased target to " << targetMs_ << "ms\n";
        }
    }
}