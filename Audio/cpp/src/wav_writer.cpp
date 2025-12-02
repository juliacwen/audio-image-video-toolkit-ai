/**
 * @file wav_writer.cpp
 * @brief wav file writer supports 16/24/32-bit PCM and 32-bit float, multi-channel
 * @author Julia Wen
 * @par Revision History
 * - 11-19-2025 — Initial check-in  
 * - 12-02-2025 — Bug fixes: 24-bit conversion, error handling, overflow protection
 */

#include "../inc/wav_writer.h"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <limits>
#include <cassert>

template <typename T>
void WavWriter::writeLE(std::ofstream& f, T value) {
    for (size_t i = 0; i < sizeof(T); ++i) {
        f.put(static_cast<char>((static_cast<uint64_t>(value) >> (8 * i)) & 0xFF));
    }
}

// Specialization for float 
template <>
void WavWriter::writeLE<float>(std::ofstream& f, float value) {
    // Reinterpret float as bytes and write little-endian
    uint32_t intRep;
    std::memcpy(&intRep, &value, sizeof(float));
    for (size_t i = 0; i < sizeof(float); ++i) {
        f.put(static_cast<char>((intRep >> (8 * i)) & 0xFF));
    }
}

// Helper: write 3 bytes little-endian (24-bit)
void WavWriter::writeInt24(int32_t v) {
    // Clamp to 24-bit signed range
    v = std::clamp(v, PCM24_MIN, PCM24_MAX);
    
    uint8_t bytes[3];
    bytes[0] = static_cast<uint8_t>(v & 0xFF);
    bytes[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    bytes[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    file_.write(reinterpret_cast<char*>(bytes), 3);
    
    if (!file_.good()) {
        throw std::runtime_error("Failed to write 24-bit sample");
    }
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
    if (sampleRate_ <= 0) {
        throw std::runtime_error("sampleRate must be > 0");
    }
    file_.open(filename, std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open WAV file: " + filename);
    }
    writeHeader();
}

// Destructor
WavWriter::~WavWriter() {
    if (!finalized_ && file_.is_open()) {
        try {
            finalize();
        } catch (...) {
            // Suppress exceptions in destructor
        }
        file_.close();
    }
}

// Move constructor
WavWriter::WavWriter(WavWriter&& other) noexcept
    : file_(std::move(other.file_)),
      sampleRate_(other.sampleRate_),
      numChannels_(other.numChannels_),
      bitDepth_(other.bitDepth_),
      dataBytesWritten_(other.dataBytesWritten_),
      finalized_(other.finalized_)
{
    other.dataBytesWritten_ = 0;
    other.sampleRate_ = 0;
    other.numChannels_ = 0;
    other.bitDepth_ = 0;
    other.finalized_ = true;
}

// Move assignment
WavWriter& WavWriter::operator=(WavWriter&& other) noexcept {
    if (this != &other) {
        if (file_.is_open() && !finalized_) {
            try {
                finalize();
            } catch (...) {
                // Suppress exceptions in noexcept function
            }
            file_.close();
        }
        file_ = std::move(other.file_);
        sampleRate_ = other.sampleRate_;
        numChannels_ = other.numChannels_;
        bitDepth_ = other.bitDepth_;
        dataBytesWritten_ = other.dataBytesWritten_;
        finalized_ = other.finalized_;

        other.dataBytesWritten_ = 0;
        other.sampleRate_ = 0;
        other.numChannels_ = 0;
        other.bitDepth_ = 0;
        other.finalized_ = true;
    }
    return *this;
}

bool WavWriter::isOpen() const {
    return file_.is_open();
}

int WavWriter::getNumChannels() const {
    return numChannels_;
}

int WavWriter::getSampleRate() const {
    return sampleRate_;
}

int WavWriter::getBitDepth() const {
    return bitDepth_;
}

void WavWriter::flush() {
    if (file_.is_open()) {
        file_.flush();
    }
}

void WavWriter::writeHeader() {
    // "RIFF"
    file_.write("RIFF", 4);
    writeLE<uint32_t>(file_, static_cast<uint32_t>(RIFF_CHUNK_SIZE_PLACEHOLDER));
    file_.write("WAVE", 4);

    // "fmt "
    file_.write("fmt ", 4);
    writeLE<uint32_t>(file_, static_cast<uint32_t>(FMT_SUBCHUNK_SIZE));

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
    writeLE<uint32_t>(file_, static_cast<uint32_t>(DATA_PLACEHOLDER));

    if (!file_.good()) {
        throw std::runtime_error("Failed to write WAV header");
    }
}

void WavWriter::finalize() {
    if (!file_.is_open()) {
        return;
    }
    std::streamoff fileSize = file_.tellp();
    if (fileSize < 0) {
        throw std::runtime_error("Failed to get file size during finalization");
    }
    uint32_t riffSize = static_cast<uint32_t>(fileSize - 8);
    uint32_t dataSize = static_cast<uint32_t>(fileSize - WAV_HEADER_SIZE);

    // Update RIFF chunk size
    file_.seekp(RIFF_CHUNK_SIZE_OFFSET);
    if (!file_.good()) {
        throw std::runtime_error("Failed to seek to RIFF chunk size offset");
    }
    writeLE<uint32_t>(file_, riffSize);

    // Update data chunk size
    file_.seekp(DATA_CHUNK_SIZE_OFFSET);
    if (!file_.good()) {
        throw std::runtime_error("Failed to seek to data chunk size offset");
    }
    writeLE<uint32_t>(file_, dataSize);

    file_.seekp(0, std::ios::end);
    file_.flush();
}

// Write single sample overloads
void WavWriter::writeSample(int16_t sample) {
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot write sample: file is not open");
    }

    if (bitDepth_ == 32) {
        // Convert int16 to float [-1.0, 1.0]
        float f = static_cast<float>(sample) / PCM16_SCALE;
        writeLE<float>(file_, f);
        dataBytesWritten_ += sizeof(float);
    } else if (bitDepth_ == 24) {
        // Scale 16-bit to 24-bit range (shift left by 8 bits)
        int32_t v = static_cast<int32_t>(sample) << 8;
        writeInt24(v);
    } else { // 16-bit
        writeLE<int16_t>(file_, sample);
        dataBytesWritten_ += sizeof(int16_t);
    }

    if (!file_.good()) {
        throw std::runtime_error("Failed to write int16 sample");
    }
}

void WavWriter::writeSample(int32_t sample) {
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot write sample: file is not open");
    }

    if (bitDepth_ == 32) {
        float f = (sample == std::numeric_limits<int32_t>::min()) ?
                  -1.0f :
                  static_cast<float>(sample) / static_cast<float>(std::numeric_limits<int32_t>::max());
        writeLE<float>(file_, f);
        dataBytesWritten_ += sizeof(float);
    } else if (bitDepth_ == 24) {
        // Use lower 24 bits as signed 24-bit sample
        writeInt24(sample);
    } else { // 16-bit target
        // Use upper 16 bits (assuming int32 is left-aligned 32-bit sample)
        int16_t s16 = static_cast<int16_t>(sample >> 16);
        writeLE<int16_t>(file_, s16);
        dataBytesWritten_ += sizeof(int16_t);
    }

    if (!file_.good()) {
        throw std::runtime_error("Failed to write int32 sample");
    }
}

void WavWriter::writeSample(float sample) {
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot write sample: file is not open");
    }

    float clamped = std::clamp(sample, -1.0f, 1.0f);

    if (bitDepth_ == 32) {
        writeLE<float>(file_, clamped);
        dataBytesWritten_ += sizeof(float);
    } else if (bitDepth_ == 24) {
        // Convert to 24-bit PCM with proper clamping
        int32_t v;
        if (clamped >= 1.0f) {
            v = PCM24_MAX;
        } else if (clamped <= -1.0f) {
            v = PCM24_MIN;
        } else {
            v = static_cast<int32_t>(clamped * PCM24_SCALE);
        }
        writeInt24(v);
    } else { // 16-bit
        // Convert to 16-bit PCM with proper clamping
        int16_t v;
        if (clamped >= 1.0f) {
            v = std::numeric_limits<int16_t>::max();
        } else if (clamped <= -1.0f) {
            v = std::numeric_limits<int16_t>::min();
        } else {
            v = static_cast<int16_t>(clamped * PCM16_SCALE);
        }
        writeLE<int16_t>(file_, v);
        dataBytesWritten_ += sizeof(int16_t);
    }

    if (!file_.good()) {
        throw std::runtime_error("Failed to write float sample");
    }
}

// Arrays
void WavWriter::writeSamples(const int32_t* samples, size_t count) {
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot write samples: file is not open");
    }
    if (samples == nullptr) {
        throw std::invalid_argument("samples pointer is null");
    }

    if (bitDepth_ == 32) {
        // Convert each to float for 32-bit float WAV
        for (size_t i = 0; i < count; ++i) {
            writeSample(samples[i]);
        }
        return;
    }

    // Bulk batch for 16/24-bit for performance
    for (size_t i = 0; i < count; ++i) {
        writeSample(samples[i]);
    }
}

void WavWriter::writeSamples(const float* samples, size_t count, bool clamp) {
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot write samples: file is not open");
    }
    if (samples == nullptr) {
        throw std::invalid_argument("samples pointer is null");
    }

    if (bitDepth_ == 32) {
        if (clamp) {
            for (size_t i = 0; i < count; ++i) {
                float clamped = std::clamp(samples[i], -1.0f, 1.0f);
                writeLE<float>(file_, clamped);
            }
        } else {
            file_.write(reinterpret_cast<const char*>(samples), count * sizeof(float));
        }
        dataBytesWritten_ += static_cast<uint32_t>(count * sizeof(float));
        if (!file_.good()) {
            throw std::runtime_error("Failed to write float samples");
        }
        return;
    }

    // For 16/24-bit, convert per sample
    for (size_t i = 0; i < count; ++i) {
        writeSample(samples[i]);
    }
}

void WavWriter::writeFrame(const float* frame) {
    writeFrame(frame, static_cast<size_t>(numChannels_));
}

void WavWriter::writeFrame(const int32_t* frame) {
    writeFrame(frame, static_cast<size_t>(numChannels_));
}

// Frames (interleaved frame pointer)
void WavWriter::writeFrame(const float* frame, size_t frameLength) {
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot write frame: file is not open");
    }
    if (frame == nullptr) {
        throw std::invalid_argument("frame pointer is null");
    }
    if (frameLength < static_cast<size_t>(numChannels_)) {
        throw std::invalid_argument("frame pointer length less than numChannels");
    }

    for (int ch = 0; ch < numChannels_; ++ch) {
        writeSample(frame[ch]);
    }
}

void WavWriter::writeFrame(const int32_t* frame, size_t frameLength) {
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot write frame: file is not open");
    }
    if (frame == nullptr) {
        throw std::invalid_argument("frame pointer is null");
    }
    if (frameLength < static_cast<size_t>(numChannels_)) {
        throw std::invalid_argument("frame pointer length less than numChannels");
    }

    for (int ch = 0; ch < numChannels_; ++ch) {
        writeSample(frame[ch]);
    }
}
