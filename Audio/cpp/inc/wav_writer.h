

/* wav_writer.h
 * WavWriter: supports 16/24/32-bit PCM and 32-bit float, multi-channel
 * @author Julia Wen (wendigilane@gmail.com)
 * @date 2025-11-19
 */

#pragma once
#include <fstream>
#include <string>
#include <cstdint>
#include <cstddef>

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

    // Query
    bool isOpen() const;
    int getNumChannels() const;

    // Write single samples (convenience overloads)
    void writeSample(int16_t sample);   // 16-bit input
    void writeSample(int32_t sample);   // int32 used for 24-bit raw or packed ints
    void writeSample(float sample);     // float input

    // Write arrays (interleaved if multi-channel)
    void writeSamples(const int32_t* samples, size_t count); // count = num samples (not frames)
    void writeSamples(const float* samples, size_t count);

    // Write one interleaved frame (frame pointer length >= numChannels)
    void writeFrame(const float* frame);   // frame of floats (per-channel)
    void writeFrame(const int32_t* frame); // frame of ints (per-channel)

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

private:
    void writeHeader();
    void finalize();

    template <typename T>
    static void writeLE(std::ofstream& f, T value);

    void writeInt24(int32_t v);

private:
    std::ofstream file_;
    int sampleRate_;
    int numChannels_;
    int bitDepth_;
    uint32_t dataBytesWritten_ = 0;
};
