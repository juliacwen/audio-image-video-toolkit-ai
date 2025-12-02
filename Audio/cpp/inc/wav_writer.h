/* wav_writer.h
 * WavWriter: supports 16/24/32-bit PCM and 32-bit float, multi-channel
 * @author Julia Wen (wendigilane@gmail.com)
 *
 * @par Revision History
 * - 11-19-2025 — Initial check-in  
 * - 12-02-2025 — Bug fixes and improvements
 */

#pragma once
#include <fstream>
#include <string>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <cassert>

class WavWriter {
public:
    // Constructor / Destructor
    WavWriter(const std::string& filename, int sampleRate, int numChannels, int bitDepth = 16);
    ~WavWriter();

    // Deleted copy (ofstream is non-copyable)
    WavWriter(const WavWriter&) = delete;
    WavWriter& operator=(const WavWriter&) = delete;

    // Move semantics
    WavWriter(WavWriter&& other) noexcept;
    WavWriter& operator=(WavWriter&& other) noexcept;

    // Query / State control
    bool isOpen() const;
    int getNumChannels() const;
    int getSampleRate() const;
    int getBitDepth() const;

    // Close file explicitly and finalize, throws on error
    void close();

    // Write single samples (convenience overloads)
    void writeSample(int16_t sample);
    void writeSample(int32_t sample);
    void writeSample(float sample);

    // Write arrays (interleaved if multi-channel)
    void writeSamples(const int32_t* samples, size_t count);
    void writeSamples(const float* samples, size_t count, bool clamp = false);

    // Backwards-compatible: assumes frameLength == numChannels_
    void writeFrame(const float* frame);  
    void writeFrame(const int32_t* frame); 

    // New safer version with explicit length check
    // Write one interleaved frame (frame pointer length >= numChannels)
    void writeFrame(const float* frame, size_t frameLength);
    void writeFrame(const int32_t* frame, size_t frameLength);

    // Explicitly flush buffered data to disk
    void flush();

    // WAV format constants (public)
    static constexpr int16_t AUDIO_FORMAT_PCM   = 1;
    static constexpr int16_t AUDIO_FORMAT_FLOAT = 3;

    static constexpr std::streamoff RIFF_CHUNK_SIZE_OFFSET = 4;
    static constexpr std::streamoff DATA_CHUNK_SIZE_OFFSET = 40;
    static constexpr std::streamoff WAV_HEADER_SIZE        = 44;

    static constexpr int32_t FMT_SUBCHUNK_SIZE = 16;
    static constexpr int32_t DATA_PLACEHOLDER  = 0;
    static constexpr int32_t RIFF_CHUNK_SIZE_PLACEHOLDER = 36;

    static constexpr float PCM16_SCALE = 32767.0f;
    static constexpr float PCM24_SCALE = 8388607.0f;

    static constexpr int32_t PCM24_MAX = 8388607;   // 2^23 - 1
    static constexpr int32_t PCM24_MIN = -8388608;  // -2^23

private:
    void writeHeader();
    void finalize();

    template <typename T>
    static void writeLE(std::ofstream& f, T value);

    // Write signed 24-bit little-endian, using low 24 bits of v
    void writeInt24(int32_t v);

private:
    std::ofstream file_;
    int sampleRate_ = 0;
    int numChannels_ = 0;
    int bitDepth_ = 0;
    uint32_t dataBytesWritten_ = 0;
    bool finalized_ = false;
};
