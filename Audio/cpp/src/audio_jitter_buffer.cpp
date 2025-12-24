/*
 * @file audio_jitter_buffer.cpp
 * @brief RTP Jitter Buffer implementation
 * @author Julia Wen (wendigilane@gmail.com)
 * 
 * This jitter buffer smooths out network jitter by buffering incoming RTP packets
 * and feeding them to the audio output at a steady rate. It handles:
 * - Packet loss detection and concealment (silence insertion)
 * - Sequence number wraparound (uint16_t wraps at 65536)
 * - Adaptive buffer sizing based on network conditions
 * - Duplicate packet detection
 * 
 * @par Revision History
 * - 12-23-2025 — Initial Checkin
 * - 12-24-2025 — Fix sequence number wraparound handling
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include "../inc/audio_jitter_buffer.h"
#include "../inc/denoise_config.h"

/**
 * @brief Constructor - Initialize jitter buffer with target latency
 * @param sampleRate Audio sample rate (e.g., 48000 Hz)
 * @param channels Number of audio channels (1=mono, 2=stereo)
 * @param targetMs Target buffering latency in milliseconds (default 80ms)
 */
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

/**
 * @brief Add incoming RTP packet to jitter buffer
 * @param packet RTP packet containing audio data and sequence number
 * @return true if packet accepted, false if rejected (duplicate)
 * 
 * Thread-safe: Protected by mutex
 */
bool AudioJitterBuffer::addPacket(const RTPPacket& packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check for duplicate - Handle sequence number wraparound properly
    // RTP sequence is uint16_t (0-65535), must handle wrap from 65535→0
    if (!firstPacket_) {
        // Cast to int16_t for proper signed wraparound arithmetic
        // Example: seq=0 after seq=65535 gives diff=(0-65535)=1 (not -65535)
        int32_t seqDiff = static_cast<int32_t>(static_cast<int16_t>(packet.header.sequenceNumber - lastSeq_));
        if (seqDiff <= 0) {
            // Duplicate or out-of-order old packet - reject it
            stats_.duplicates++;
            return false;
        }
    }
    
    // Detect packet loss - Handle wraparound properly
    if (!firstPacket_) {
        uint16_t expected = (lastSeq_ + 1) & 0xFFFF;  // Next expected sequence number
        
        // Detect sequence wraparound: lastSeq in upper range (e.g., 65000-65535), current in lower range (e.g., 0-1000)
        // This handles wraparound even with packet loss
        if (lastSeq_ > 65000 && packet.header.sequenceNumber < 1000) {
            std::cout << "[AudioJitterBuffer] RTP sequence wraparound: " 
                      << lastSeq_ << " → " << packet.header.sequenceNumber
                      << " (total packets: " << stats_.packetsReceived + 1 << ")\n";
        }
        
        if (packet.header.sequenceNumber != expected) {
            // Gap detected - calculate how many packets were lost
            int32_t lost = static_cast<int32_t>(static_cast<int16_t>(packet.header.sequenceNumber - expected));
            
            if (lost > 0 && lost < 1000) {  
                // Reasonable gap - insert silence for each lost packet
                // Sanity check prevents inserting thousands of silence frames on clock jumps
                stats_.packetsLost += lost;
                
                for (int32_t i = 0; i < lost; i++) {
                    insertSilence();  // Add one frame of silence per lost packet
                }
            } else if (lost > 0) {
                // Huge gap detected - likely sender restart or clock jump
                std::cerr << "[AudioJitterBuffer] Detected " << lost 
                          << " lost packets - possible clock jump or restart\n";
                stats_.packetsLost += lost;
                // Don't insert silence for huge gaps to avoid memory issues
            }
        }
    }
    
    // Update last seen sequence number
    lastSeq_ = packet.header.sequenceNumber;
    firstPacket_ = false;
    
    // Add packet to buffer (FIFO queue)
    packets_.push_back(packet);
    stats_.packetsReceived++;
    
    // Start playing when buffer reaches target level (initial buffering)
    if (!playing_ && getBufferedSamples() >= targetSamples_) {
        playing_ = true;
        static bool printed = false;
        if (!printed) {
            std::cout << "[AudioJitterBuffer] Started playback (buffered " 
                      << getBufferedSamples() << " samples, " << targetMs_ << "ms)\n";
            printed = true;
        }
    }
    
    // Prevent buffer overflow - drop oldest packets if too full
    // This prevents unbounded memory growth on network issues
    if (getBufferedSamples() > maxSamples_) {
        if (!packets_.empty()) {
            packets_.pop_front();  // Drop oldest packet
            stats_.overflow++;
        }
    }
    
    return true;
}

/**
 * @brief Get samples from jitter buffer for audio playback
 * @param output Output buffer to fill with audio samples
 * @param requestedSamples Number of samples requested (typically one frame)
 * @return Number of samples actually retrieved (always equals requestedSamples)
 * 
 * Thread-safe: Protected by mutex
 * Note: Always returns requestedSamples (fills with silence if needed)
 */
size_t AudioJitterBuffer::getSamples(float* output, size_t requestedSamples) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!playing_) {
        // Still buffering - not enough data to start playback yet
        // Return silence and count as underrun only if we've received packets
        std::fill(output, output + requestedSamples, 0.0f);
        if (stats_.packetsReceived > 0) {
            stats_.underruns++;
        }
        return requestedSamples;
    }
    
    size_t samplesRetrieved = 0;
    
    // Extract samples from buffered packets
    while (samplesRetrieved < requestedSamples && !packets_.empty()) {
        RTPPacket& packet = packets_.front();
        size_t packetSamples = packet.payload.size() / sizeof(float);
        size_t samplesNeeded = requestedSamples - samplesRetrieved;
        size_t samplesToCopy = std::min(packetSamples, samplesNeeded);
        
        // Copy samples from packet payload to output
        std::memcpy(output + samplesRetrieved, 
                   packet.payload.data(), 
                   samplesToCopy * sizeof(float));
        
        samplesRetrieved += samplesToCopy;
        
        // Remove packet if fully consumed
        if (samplesToCopy >= packetSamples) {
            packets_.pop_front();
        } else {
            // Partial consume - remove consumed samples from packet
            // (This handles cases where request size doesn't align with packet size)
            packet.payload.erase(packet.payload.begin(), 
                                packet.payload.begin() + samplesToCopy * sizeof(float));
        }
    }
    
    // Check for underrun - buffer ran out before fulfilling request
    if (samplesRetrieved < requestedSamples) {
        // Fill remaining with silence
        std::fill(output + samplesRetrieved, 
                 output + requestedSamples, 0.0f);
        stats_.underruns++;
        
        // Reset to buffering state if buffer is completely empty
        if (packets_.empty() && playing_) {
            playing_ = false;
            // Reduce console spam - only print every 10th underrun
            if (stats_.underruns % JITTER_UNDERRUN_PRINT_INTERVAL == 0) {
                std::cout << "[AudioJitterBuffer] Underrun #" << stats_.underruns 
                          << " - rebuffering\n";
            }
        }
    }
    
    // Adaptive buffer adjustment based on network conditions
    adaptBufferSize();
    
    return requestedSamples;
}

/**
 * @brief Get current number of samples buffered
 * @return Total samples across all packets in buffer
 * 
 * Sums up payload sizes of all packets in the queue.
 * Thread-safe: Must be called with mutex held or from const method.
 */
size_t AudioJitterBuffer::getBufferedSamples() const {
    size_t total = 0;
    for (const auto& packet : packets_) {
        total += packet.payload.size() / sizeof(float);
    }
    return total;
}

/**
 * @brief Reset jitter buffer to initial state
 * 
 * Clears all buffered packets and resets state.
 * Used when restarting stream or handling fatal errors.
 * Thread-safe: Protected by mutex
 */
void AudioJitterBuffer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    packets_.clear();
    playing_ = false;
    firstPacket_ = true;
    lastSeq_ = 0;
}

/**
 * @brief Get statistics snapshot
 * @return Copy of current statistics
 * 
 * Thread-safe: Protected by mutex, returns copy
 */
AudioJitterBuffer::Statistics AudioJitterBuffer::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

/**
 * @brief Get current buffering latency in milliseconds
 * @return Latency in ms (0 if buffer empty)
 * 
 * Calculates: (buffered_samples / sample_rate / channels) * 1000
 * Example: 7680 samples / 48000 Hz / 2 channels * 1000 = 80ms
 * Thread-safe: Protected by mutex
 */
int AudioJitterBuffer::getCurrentLatencyMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t buffered = getBufferedSamples();
    if (buffered == 0) return 0;
    return (buffered * MS_PER_SECOND) / (sampleRate_ * channels_);
}

/**
 * @brief Insert one frame of silence (packet loss concealment)
 * 
 * Creates a dummy packet filled with zeros to replace lost packet.
 * This maintains timing and prevents audio gaps from turning into long silences.
 * Called when packet loss is detected. Must be called with mutex held.
 */
void AudioJitterBuffer::insertSilence() {
    RTPPacket silencePacket;
    size_t silenceSize = FRAME_SIZE * channels_ * sizeof(float);
    silencePacket.payload.resize(silenceSize, 0);  // Fill with zeros
    packets_.push_back(silencePacket);
}

/**
 * @brief Adapt buffer target size based on network conditions
 * 
 * Increases buffer if experiencing frequent underruns (network jitter/loss)
 * Decreases buffer if consistently too full (wastes latency)
 * 
 * Trade-off: Larger buffer = more latency but fewer underruns
 * 
 * Must be called with mutex held.
 */
void AudioJitterBuffer::adaptBufferSize() {
    // Increase buffer if experiencing underruns
    // Less aggressive - only adjust every N underruns to avoid thrashing
    if (stats_.underruns > lastUnderruns_ + JITTER_UNDERRUN_THRESHOLD) {
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
    
    // Decrease buffer if consistently overfull
    // Only decrease if we have very consistent good performance
    if (buffered > targetSamples_ * 2 && stats_.overflow > lastOverflow_ + JITTER_OVERFLOW_THRESHOLD) {
        int newTarget = std::max(targetMs_ - JITTER_DECREASE_STEP_MS, JITTER_BUFFER_MIN_MS);
        if (newTarget != targetMs_) {
            targetMs_ = newTarget;
            targetSamples_ = (sampleRate_ * targetMs_) / MS_PER_SECOND * channels_;
            lastOverflow_ = stats_.overflow;
            std::cout << "[AudioJitterBuffer] Decreased target to " << targetMs_ << "ms\n";
        }
    }
}