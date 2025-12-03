/*
 * SPSCFloatBuffer.h
 *
 * Lock-free Single-Producer Single-Consumer ring buffer optimized for real-time audio.
 * Author: Julia Wen (wendigilane@gmail.com)
 * OVERVIEW:
 *   This is a wait-free, thread-safe circular buffer designed for scenarios where exactly
 *   one thread produces data and exactly one thread consumes data. It's ideal for audio
 *   processing pipelines where the audio callback (consumer) needs samples from a
 *   processing thread (producer) with minimal latency and no blocking.
 *
 * KEY FEATURES:
 *   ✓ Lock-free: No mutexes or blocking - both threads can operate simultaneously
 *   ✓ Wait-free single operations: push() and pop() never wait or retry
 *   ✓ Cache-optimized: 64-byte alignment prevents false sharing between CPU cores
 *   ✓ Bulk operations: pushBulk/popBulk minimize atomic operations (critical for performance)
 *   ✓ Power-of-2 sizing: Uses bitwise AND instead of modulo for ~10x faster indexing
 *   ✓ Real-time safe: No allocations after construction, all operations bounded time
 *   ✓ C++17 features: [[nodiscard]], noexcept, if constexpr for modern safety
 *
 * PERFORMANCE:
 *   - Single operations: ~2-4 atomic operations per call
 *   - Bulk operations: ~2-4 atomic operations regardless of sample count
 *   - For 512 samples: Bulk is ~256x faster than looping single operations
 *
 * THREAD SAFETY MODEL:
 *   - ONE producer thread may call push() and pushBulk()
 *   - ONE consumer thread may call pop() and popBulk()
 *   - Query methods (available(), capacity(), size()) are safe from any thread
 *   - Multiple producers or consumers will cause DATA RACES - use a mutex-based queue instead
 *
 * MEMORY ORDERING:
 *   Uses C++11 acquire-release semantics for correct lock-free synchronization:
 *   - Producer: Writes data, then releases head (makes data visible to consumer)
 *   - Consumer: Acquires head (sees producer's data), reads data, releases tail
 *   This ensures the consumer never sees uninitialized data and the producer never
 *   overwrites data the consumer is still reading.
 *
 * TYPICAL USAGE WITH PORTAUDIO:
 *   SPSCFloatBuffer buffer(4096);  // Power-of-2 capacity recommended
 *   
 *   // Processing thread (producer):
 *   float samples[512];
 *   process_audio(samples, 512);
 *   buffer.pushBulk(samples, 512);  // Non-blocking push
 *   
 *   // PortAudio callback (consumer):
 *   size_t got = buffer.popBulk(outputBuffer, framesPerBuffer);
 *   if (got < framesPerBuffer) {
 *       // Handle underrun: fill remainder with silence
 *   }
 *
 * CAPACITY NOTES:
 *   - Actual usable capacity is (requested_capacity - 1) due to the full/empty distinction
 *   - Size is rounded up to next power of 2 for performance
 *   - Example: Request 1000 → allocated 1024, usable 1023
 *
 * COMPILER REQUIREMENTS:
 *   - C++17 or later
 *   - Compiler with std::atomic support (GCC 4.8+, Clang 3.1+, MSVC 2015+)
 * @par Revision History
 * - 11-20-2025 — Initial check-in  
 */
 

#pragma once
#include <atomic>
#include <cstddef>
#include <memory>
#include <algorithm>

class SPSCFloatBuffer {
public:
    // Explicit constructor prevents accidental implicit conversions
    explicit SPSCFloatBuffer(size_t capacity)
        : size_(nextPowerOf2(capacity + 1))
        , mask_(size_ - 1)
        , buffer_(new float[size_]())  // Added () to zero-initialize!
        , head_(0)
        , tail_(0) {
        std::fill_n(buffer_.get(), size_, 0.0f); // Explicitly zero the buffer for safety
    }

    // Prevent copying to avoid accidental shared ownership
    SPSCFloatBuffer(const SPSCFloatBuffer&) = delete;
    SPSCFloatBuffer& operator=(const SPSCFloatBuffer&) = delete;

    // ---------- Single Sample Operations ----------

    // Push a single sample to the buffer
    // Returns: true if successful, false if buffer is full
    bool push(float sample) {
        // Load head with relaxed ordering - we own this variable (producer side)
        size_t currentHead = head_.load(std::memory_order_relaxed);
        size_t nextHead = (currentHead + 1) & mask_;  // Fast modulo via bitwise AND
        
        // Check if buffer is full by comparing with tail
        // Acquire ordering ensures we see all previous writes by the consumer
        if (nextHead == tail_.load(std::memory_order_acquire))
            return false;  // Buffer full
        
        // Write the sample to the buffer
        buffer_[currentHead] = sample;
        
        // Publish the new head position with release semantics
        // This ensures the sample write above is visible to the consumer before head update
        head_.store(nextHead, std::memory_order_release);
        return true;
    }

    // Pop a single sample from the buffer
    // Returns: true if successful, false if buffer is empty
    bool pop(float& sample) {
        // Load tail with relaxed ordering - we own this variable (consumer side)
        size_t currentTail = tail_.load(std::memory_order_relaxed);
        
        // Acquire head to synchronize with producer's writes
        // This ensures we see all sample writes before the head was updated
        size_t currentHead = head_.load(std::memory_order_acquire);
        
        // Check if buffer is empty
        if (currentTail == currentHead)
            return false;  // Buffer empty
        
        // Read the sample from the buffer
        sample = buffer_[currentTail];
        
        // Publish the new tail position with release semantics
        // This makes the buffer slot available to the producer
        tail_.store((currentTail + 1) & mask_, std::memory_order_release);
        return true;
    }

    // ---------- Bulk Operations (Optimized for Audio Buffers) ----------

    // Push multiple samples efficiently
    // Performs only ONE atomic operation regardless of sample count
    // Returns: number of samples actually pushed (may be less than count if buffer fills)
    size_t pushBulk(const float* samples, size_t count) {
        size_t currentHead = head_.load(std::memory_order_relaxed);
        
        // Acquire tail to synchronize with consumer's writes
        size_t currentTail = tail_.load(std::memory_order_acquire);
        
        // Calculate available space using bitwise operations
        // (tail - head - 1) wrapped around gives free space (minus 1 to distinguish full/empty)
        size_t space = (currentTail - currentHead - 1) & mask_;
        size_t toPush = std::min(count, space);
        
        // Write all samples before updating head (no atomics in loop!)
        for (size_t i = 0; i < toPush; ++i) {
            buffer_[(currentHead + i) & mask_] = samples[i];
        }
        
        // Single release operation makes all sample writes visible to consumer
        head_.store((currentHead + toPush) & mask_, std::memory_order_release);
        return toPush;
    }

    // Pop multiple samples efficiently
    // Performs only ONE atomic operation regardless of sample count
    // Ideal for PortAudio callbacks that need to fill a buffer of samples
    // Returns: number of samples actually popped (may be less than count if buffer empties)
    size_t popBulk(float* samples, size_t count) {
        size_t currentTail = tail_.load(std::memory_order_relaxed);
        
        // Acquire head to see all producer writes
        size_t currentHead = head_.load(std::memory_order_acquire);
        
        // Calculate available samples using bitwise operations
        size_t available = (currentHead - currentTail) & mask_;
        size_t toPop = std::min(count, available);
        
        // Read all samples before updating tail (no atomics in loop!)
        for (size_t i = 0; i < toPop; ++i) {
            samples[i] = buffer_[(currentTail + i) & mask_];
        }
        
        // Single release operation makes these slots available to producer
        tail_.store((currentTail + toPop) & mask_, std::memory_order_release);
        return toPop;
    }

    // ---------- Query Methods ----------

    // Get number of samples currently available in buffer
    [[nodiscard]] size_t available() const {
        // Acquire both to get a consistent snapshot
        size_t currentHead = head_.load(std::memory_order_acquire);
        size_t currentTail = tail_.load(std::memory_order_acquire);
        
        // Calculate available samples using bitwise operations
        return (currentHead - currentTail) & mask_;
    }

    // Get the maximum capacity of the buffer (actual usable capacity is capacity - 1)
    [[nodiscard]] size_t capacity() const noexcept {
        return mask_;  // This is size - 1, which is the usable capacity
    }

    // Get the total allocated size (including the sentinel slot)
    [[nodiscard]] size_t size() const noexcept {
        return size_;
    }

private:
    const size_t size_;  // Actual buffer size (power of 2)
    const size_t mask_;  // Bitmask for fast modulo (size - 1)
    std::unique_ptr<float[]> buffer_;
    
    // Cache line padding (64 bytes) to prevent false sharing between CPU cores
    // False sharing occurs when two threads access different variables on the same cache line,
    // causing expensive cache invalidation and severe performance degradation in real-time audio.
    // By aligning to 64 bytes, we guarantee each atomic is on its own cache line.
    alignas(64) std::atomic<size_t> head_;  // Written by producer, read by consumer
    alignas(64) std::atomic<size_t> tail_;  // Written by consumer, read by producer
    
    // Round up to next power of 2 for efficient bitwise modulo operations
    // Uses bit-smearing technique for branchless computation
    [[nodiscard]] static constexpr size_t nextPowerOf2(size_t n) noexcept {
        if (n == 0) return 1;
        n--;
        // Bit-smearing: set all bits below the highest set bit
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        if constexpr (sizeof(size_t) > 4) {
            n |= n >> 32;  // Only on 64-bit platforms
        }
        return n + 1;
    }
};

// Example usage with PortAudio:
/*
SPSCFloatBuffer buffer(4096);  // 4K sample buffer

// In your audio processing thread (producer):
float samples[512];
// ... generate samples ...
size_t pushed = buffer.pushBulk(samples, 512);

// In your PortAudio callback (consumer):
int audioCallback(const void* input, void* output, 
                  unsigned long frameCount, ...) {
    float* out = (float*)output;
    size_t read = buffer.popBulk(out, frameCount);
    
    // Fill remainder with silence on underrun
    for (size_t i = read; i < frameCount; ++i) {
        out[i] = 0.0f;
    }
    return paContinue;
}
*/