
/*wav_writer.cpp
 * WavWriter: supports 16/24/32-bit PCM and 32-bit float, multi-channel
 * Author: Julia Wen (wendigilane@gmail.com)
 * Date: 11-19-2025
 */

#include "../inc/wav_writer.h"
#include <stdexcept>
#include <cstring>
#include <algorithm>

// Helper: write little-endian for integral types
template <typename T>
void WavWriter::writeLE(std::ofstream& f, T value) {
    for (size_t i = 0; i < sizeof(T); ++i) {
        f.put(static_cast<char>((static_cast<uint64_t>(value) >> (8 * i)) & 0xFF));
    }
}

// Helper: write 3 bytes little-endian (24-bit)
void WavWriter::writeInt24(int32_t v) {
    uint8_t bytes[3];
    bytes[0] = static_cast<uint8_t>(v & 0xFF);
    bytes[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    bytes[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    file_.write(reinterpret_cast<char*>(bytes), 3);
    dataBytesWritten_ += 3;
}

// Constructor
WavWriter::WavWriter(const std::string& filename, int sampleRate, int numChannels, int bitDepth)
    : sampleRate_(sampleRate),
      numChannels_(numChannels),
      bitDepth_(bitDepth)
{
    if (bitDepth_ != 16 && bitDepth_ != 24 && bitDepth_ != 32) {
        throw std::runtime_error("Unsupported bit depth. Use 16, 24, or 32.");
    }
    if (numChannels_ < 1) {
        throw std::runtime_error("numChannels must be >= 1");
    }

    file_.open(filename, std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open WAV file: " + filename);
    }

    writeHeader();
}

// Destructor
WavWriter::~WavWriter() {
    if (file_.is_open()) {
        finalize();
        file_.close();
    }
}

// Move constructor
WavWriter::WavWriter(WavWriter&& other) noexcept
    : file_(std::move(other.file_)),
      sampleRate_(other.sampleRate_),
      numChannels_(other.numChannels_),
      bitDepth_(other.bitDepth_),
      dataBytesWritten_(other.dataBytesWritten_)
{
    other.dataBytesWritten_ = 0;
    other.sampleRate_ = 0;
    other.numChannels_ = 0;
    other.bitDepth_ = 0;
}

// Move assignment
WavWriter& WavWriter::operator=(WavWriter&& other) noexcept {
    if (this != &other) {
        if (file_.is_open()) {
            finalize();
            file_.close();
        }
        file_ = std::move(other.file_);
        sampleRate_ = other.sampleRate_;
        numChannels_ = other.numChannels_;
        bitDepth_ = other.bitDepth_;
        dataBytesWritten_ = other.dataBytesWritten_;

        other.dataBytesWritten_ = 0;
        other.sampleRate_ = 0;
        other.numChannels_ = 0;
        other.bitDepth_ = 0;
    }
    return *this;
}

bool WavWriter::isOpen() const {
    return file_.is_open();
}

int WavWriter::getNumChannels() const {
    return numChannels_;
}

void WavWriter::writeHeader() {
    // "RIFF"
    file_.write("RIFF", 4);
    writeLE<uint32_t>(file_, static_cast<uint32_t>(RIFF_CHUNK_SIZE_PLACEHOLDER)); // placeholder
    file_.write("WAVE", 4);

    // "fmt "
    file_.write("fmt ", 4);
    writeLE<uint32_t>(file_, static_cast<uint32_t>(FMT_SUBCHUNK_SIZE)); // fmt chunk size

    uint16_t audioFormat = (bitDepth_ == 32) ? AUDIO_FORMAT_FLOAT : AUDIO_FORMAT_PCM;
    writeLE<uint16_t>(file_, audioFormat);
    writeLE<uint16_t>(file_, static_cast<uint16_t>(numChannels_));
    writeLE<uint32_t>(file_, static_cast<uint32_t>(sampleRate_));

    int bytesPerSample = (bitDepth_ / 8);
    uint32_t byteRate = static_cast<uint32_t>(sampleRate_ * numChannels_ * bytesPerSample);
    writeLE<uint32_t>(file_, byteRate);

    uint16_t blockAlign = static_cast<uint16_t>(numChannels_ * bytesPerSample);
    writeLE<uint16_t>(file_, blockAlign);

    writeLE<uint16_t>(file_, static_cast<uint16_t>(bitDepth_));

    // "data"
    file_.write("data", 4);
    writeLE<uint32_t>(file_, static_cast<uint32_t>(DATA_PLACEHOLDER)); // placeholder
}

void WavWriter::finalize() {
    std::streamoff fileSize = file_.tellp();
    uint32_t riffSize = static_cast<uint32_t>(fileSize - 8);
    uint32_t dataSize = static_cast<uint32_t>(fileSize - WAV_HEADER_SIZE);

    // Update RIFF chunk size
    file_.seekp(RIFF_CHUNK_SIZE_OFFSET);
    writeLE<uint32_t>(file_, riffSize);

    // Update data chunk size
    file_.seekp(DATA_CHUNK_SIZE_OFFSET);
    writeLE<uint32_t>(file_, dataSize);

    file_.seekp(0, std::ios::end);
}

// write single sample overloads
void WavWriter::writeSample(int16_t sample) {
    if (!file_.is_open()) return;

    // if file is float bitdepth, convert int16->float
    if (bitDepth_ == 32) {
        float f = static_cast<float>(sample) / PCM16_SCALE;
        writeLE<float>(file_, f);
        dataBytesWritten_ += sizeof(float);
    } else if (bitDepth_ == 24) {
        int32_t v = static_cast<int32_t>(sample) << 8; // align 16->24 (preserve sign)
        writeInt24(v >> 8); // write lower 24 bits (we shift back to 24 range)
        // NOTE: above approach keeps dynamic range — alternative: scale to full 24
    } else { // 16-bit
        writeLE<int16_t>(file_, sample);
        dataBytesWritten_ += sizeof(int16_t);
    }
}

void WavWriter::writeSample(int32_t sample) {
    if (!file_.is_open()) return;

    if (bitDepth_ == 32) {
        // interpret sample as int32 bit pattern of float? better treat as integer -> convert
        // We'll convert integer [-2^31,2^31-1] to float [-1,1)
        float f = static_cast<float>(sample) / static_cast<float>(INT32_MAX);
        writeLE<float>(file_, f);
        dataBytesWritten_ += sizeof(float);
    } else if (bitDepth_ == 24) {
        // assume lower 24 bits contain sample (signed), write 3 bytes
        writeInt24(sample);
    } else { // 16-bit target
        int16_t s16 = static_cast<int16_t>(sample >> 16); // downscale 32->16 by shifting
        writeLE<int16_t>(file_, s16);
        dataBytesWritten_ += sizeof(int16_t);
    }
}

void WavWriter::writeSample(float sample) {
    if (!file_.is_open()) return;

    float clamped = std::clamp(sample, -1.0f, 1.0f);

    if (bitDepth_ == 32) {
        writeLE<float>(file_, clamped);
        dataBytesWritten_ += sizeof(float);
    } else if (bitDepth_ == 24) {
        int32_t v = static_cast<int32_t>(clamped * PCM24_SCALE);
        writeInt24(v);
    } else { // 16-bit
        int16_t v = static_cast<int16_t>(clamped * PCM16_SCALE);
        writeLE<int16_t>(file_, v);
        dataBytesWritten_ += sizeof(int16_t);
    }
}

// arrays
void WavWriter::writeSamples(const int32_t* samples, size_t count) {
    if (!file_.is_open() || samples == nullptr) return;
    for (size_t i = 0; i < count; ++i) writeSample(samples[i]);
}

void WavWriter::writeSamples(const float* samples, size_t count) {
    if (!file_.is_open() || samples == nullptr) return;

    if (bitDepth_ == 32) {
        // write raw floats
        file_.write(reinterpret_cast<const char*>(samples), count * sizeof(float));
        dataBytesWritten_ += static_cast<uint32_t>(count * sizeof(float));
        return;
    }

    // else convert per sample
    for (size_t i = 0; i < count; ++i) writeSample(samples[i]);
}

// frames (interleaved frame pointer)
void WavWriter::writeFrame(const float* frame) {
    if (!file_.is_open() || frame == nullptr) return;
    for (int ch = 0; ch < numChannels_; ++ch) {
        writeSample(frame[ch]);
    }
}

void WavWriter::writeFrame(const int32_t* frame) {
    if (!file_.is_open() || frame == nullptr) return;
    for (int ch = 0; ch < numChannels_; ++ch) {
        writeSample(frame[ch]);
    }
}
