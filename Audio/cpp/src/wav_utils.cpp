/**
 * @file wav_utils.cpp
 * @brief Common utilities for WAV file processing (implementation)
 * @author Julia Wen (wendigilane@gmail.com)
 * @par Revision History
 * - 11-21-2025 — Initial check-in  
 * - 12-02-2025 — Update with improvement
 * - 12-04-2025 — Added 3-column writeRow for spectrogram support
 */

#include "../inc/wav_utils.h"
#include <algorithm>
#include <cmath>
#include <array>
#include <stdexcept>
#include "../inc/error_codes.h"

namespace wav {

// ============================================================================
// Constants
// ============================================================================

const double kPI = 3.14159265358979323846;

const uint16_t PCM_16_BPS = 16;
const uint16_t PCM_24_BPS = 24;
const uint16_t PCM_32_BPS = 32;
const uint16_t WAVE_FMT_PCM = 1;
const uint16_t WAVE_FMT_FLOAT = 3;

const size_t BYTES_PER_SAMPLE_16 = 2;
const size_t BYTES_PER_SAMPLE_32 = 4;

const int32_t SIGN_BIT_24 = 0x800000;
const int32_t SIGN_EXTEND_24 = static_cast<int32_t>(0xFF000000);

const size_t FLUSH_INTERVAL = 1000;

// ============================================================================
// WavFormat Methods
// ============================================================================

size_t WavFormat::frames() const {
    // Validate all critical fields before calculation
    if (channels == 0 || bitsPerSample == 0 || bitsPerSample % 8 != 0 || sampleRate == 0) {
        return 0;
    }
    return dataSize / (bitsPerSample / 8) / channels;
}

size_t WavFormat::bytesPerSample() const {
    return bitsPerSample / 8;
}

bool WavFormat::isValid() const {
    return format != 0 && channels > 0 && sampleRate > 0 && bitsPerSample > 0;
}

bool WavFormat::isSupportedFormat() const {
    return (format == WAVE_FMT_PCM && (bitsPerSample == PCM_16_BPS || bitsPerSample == PCM_24_BPS)) ||
           (format == WAVE_FMT_FLOAT && bitsPerSample == PCM_32_BPS);
}

// ============================================================================
// Low-level Read Functions
// ============================================================================

uint16_t readU16(std::istream& in) {
    std::array<uint8_t, 2> b;
    if (!in.read(reinterpret_cast<char*>(b.data()), BYTES_PER_SAMPLE_16)) {
        throw ERR_READ_FAILURE;
    }
    return static_cast<uint16_t>(b[0] | (b[1] << 8));
}

uint32_t readU32(std::istream& in) {
    std::array<uint8_t, 4> b;
    if (!in.read(reinterpret_cast<char*>(b.data()), BYTES_PER_SAMPLE_32)) {
        throw ERR_READ_FAILURE;
    }
    return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) | 
           (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
}

int16_t readI16(const uint8_t* p) {
    return static_cast<int16_t>(p[0] | (p[1] << 8));
}

int32_t readI24(const uint8_t* p) {
    int32_t v = (p[0] | (p[1] << 8) | (p[2] << 16));
    if (v & SIGN_BIT_24) {
        v |= SIGN_EXTEND_24;
    }
    return v;
}

// ============================================================================
// WAV Header Parsing
// ============================================================================

int parseWavHeader(std::istream& f, WavFormat& fmt) {
    // Read RIFF header
    std::array<char, 4> riff;
    if (!f.read(riff.data(), 4)) {
        std::cerr << "Invalid WAV file: failed to read RIFF header\n";
        return ERR_INVALID_HEADER;
    }
    
    if (std::string(riff.data(), 4) != "RIFF") {
        std::cerr << "Invalid WAV file: missing RIFF header\n";
        return ERR_INVALID_HEADER;
    }
    
    readU32(f); // chunk size
    
    std::array<char, 4> wave;
    if (!f.read(wave.data(), 4)) {
        std::cerr << "Invalid WAV file: failed to read WAVE header\n";
        return ERR_INVALID_HEADER;
    }
    
    if (std::string(wave.data(), 4) != "WAVE") {
        std::cerr << "Invalid WAV file: missing WAVE header\n";
        return ERR_INVALID_HEADER;
    }

    bool have_fmt = false, have_data = false;
    
    // Parse chunks
    while (f && !(have_fmt && have_data)) {
        std::array<char, 4> id;
        if (!f.read(id.data(), 4)) break;
        
        uint32_t sz = readU32(f);
        std::string sid(id.data(), 4);
        
        if (sid == "fmt ") {
            have_fmt = true;
            fmt.format = readU16(f);
            fmt.channels = readU16(f);
            fmt.sampleRate = readU32(f);
            readU32(f);  // byte rate
            readU16(f);  // block align
            fmt.bitsPerSample = readU16(f);
            
            // Skip any extra format bytes
            if (sz > 16) {
                f.seekg(sz - 16, std::ios::cur);
            }
        } 
        else if (sid == "data") {
            have_data = true;
            fmt.dataPos = f.tellg();
            fmt.dataSize = sz;
            
            // Skip data chunk (and padding byte if odd size)
            f.seekg(sz + (sz & 1), std::ios::cur);
        } 
        else {
            // Skip unknown chunk (and padding byte if odd size)
            f.seekg(sz + (sz & 1), std::ios::cur);
        }
    }
    
    if (!have_fmt || !have_data) {
        std::cerr << "Invalid WAV file: missing fmt or data chunk\n";
        return ERR_INVALID_HEADER;
    }
    
    if (!fmt.isSupportedFormat()) {
        std::cerr << "Unsupported format. Supported: PCM 16/24-bit, Float 32-bit\n";
        return ERR_UNSUPPORTED_FORMAT;
    }
    
    if (fmt.sampleRate == 0 || fmt.sampleRate > 384000) {
        std::cerr << "Invalid sample rate: " << fmt.sampleRate << " Hz\n";
        return ERR_INVALID_INPUT;
    }
    
    if (fmt.channels == 0) {
        std::cerr << "Invalid number of channels: " << fmt.channels << "\n";
        return ERR_INVALID_INPUT;
    }
    
    return SUCCESS;
}

// ============================================================================
// Sample Decoding
// ============================================================================

double decodeSample(const uint8_t* ptr, const WavFormat& fmt) {
    if (fmt.format == WAVE_FMT_PCM && fmt.bitsPerSample == PCM_16_BPS) {
        return static_cast<double>(readI16(ptr));
    }
    else if (fmt.format == WAVE_FMT_PCM && fmt.bitsPerSample == PCM_24_BPS) {
        return static_cast<double>(readI24(ptr));
    }
    else if (fmt.format == WAVE_FMT_FLOAT && fmt.bitsPerSample == PCM_32_BPS) {
        return static_cast<double>(*reinterpret_cast<const float*>(ptr));
    }
    return 0.0;
}

double decodeSampleMono(const std::vector<uint8_t>& raw, size_t frame, 
                        const WavFormat& fmt) {
    const size_t sampleBytes = fmt.bytesPerSample();
    const size_t offset = frame * fmt.channels * sampleBytes;
    
    // Bounds check
    if (offset + sampleBytes > raw.size()) {
        throw std::out_of_range("Frame index out of range in decodeSampleMono");
    }
    
    const uint8_t* ptr = &raw[offset];
    
    if (fmt.channels == 1) {
        return decodeSample(ptr, fmt);
    } else {
        // Mix first two channels to mono
        double L = decodeSample(ptr, fmt);
        double R = decodeSample(ptr + sampleBytes, fmt);
        return (L + R) / 2.0;
    }
}

double decodeSampleChannel(const std::vector<uint8_t>& raw, size_t frame, 
                           size_t channel, const WavFormat& fmt) {
    // Bounds checks
    if (channel >= fmt.channels) {
        throw std::out_of_range("Channel index out of range");
    }
    
    const size_t sampleBytes = fmt.bytesPerSample();
    const size_t offset = frame * fmt.channels * sampleBytes + channel * sampleBytes;
    
    if (offset + sampleBytes > raw.size()) {
        throw std::out_of_range("Frame index out of range in decodeSampleChannel");
    }
    
    const uint8_t* ptr = &raw[offset];
    return decodeSample(ptr, fmt);
}

// ============================================================================
// Window Functions
// ============================================================================

double windowValue(WindowType type, size_t i, size_t N) {
    switch(type) {
        case WindowType::Rectangular: 
            return 1.0;
        case WindowType::Hann:
            return 0.5 * (1.0 - std::cos(2.0 * kPI * static_cast<double>(i) / static_cast<double>(N - 1)));
        case WindowType::Hamming:
            return 0.54 - 0.46 * std::cos(2.0 * kPI * static_cast<double>(i) / static_cast<double>(N - 1));
        case WindowType::Blackman:
            return 0.42 - 0.5 * std::cos(2.0 * kPI * static_cast<double>(i) / static_cast<double>(N - 1)) + 
                   0.08 * std::cos(4.0 * kPI * static_cast<double>(i) / static_cast<double>(N - 1));
    }
    return 1.0;
}

WindowType parseWindow(const std::string& w) {
    std::string s = w;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    
    if (s == "hann") return WindowType::Hann;
    if (s == "hamming") return WindowType::Hamming;
    if (s == "blackman") return WindowType::Blackman;
    
    return WindowType::Rectangular;
}

std::vector<double> generateWindowCoeffs(WindowType type, size_t N) {
    std::vector<double> coeffs(N);
    for (size_t i = 0; i < N; ++i) {
        coeffs[i] = windowValue(type, i, N);
    }
    return coeffs;
}

// ============================================================================
// CSV Writing Utilities
// ============================================================================

CsvWriter::CsvWriter(const std::string& path) : lineCount_(0) {
    file_.open(path);
    if (!file_) {
        throw ERR_CSV_WRITE_FAILURE;
    }
}

CsvWriter::~CsvWriter() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

void CsvWriter::writeHeader(const std::string& header) {
    file_ << header << "\n";
}

void CsvWriter::writeLine(const std::string& line) {
    file_ << line << "\n";
    if (++lineCount_ >= FLUSH_INTERVAL) {
        file_.flush();
        lineCount_ = 0;
    }
}

template<typename T1, typename T2>
void CsvWriter::writeRow(T1 col1, T2 col2) {
    file_ << col1 << "," << col2 << "\n";
    if (++lineCount_ >= FLUSH_INTERVAL) {
        file_.flush();
        lineCount_ = 0;
    }
}

template<typename T1, typename T2, typename T3>
void CsvWriter::writeRow(T1 col1, T2 col2, T3 col3) {
    file_ << col1 << "," << col2 << "," << col3 << "\n";
    if (++lineCount_ >= FLUSH_INTERVAL) {
        file_.flush();
        lineCount_ = 0;
    }
}

// Explicit template instantiations for common types
template void CsvWriter::writeRow<size_t, double>(size_t, double);
template void CsvWriter::writeRow<double, double>(double, double);
template void CsvWriter::writeRow<int, double>(int, double);

// 3-column instantiations for spectrogram support
template void CsvWriter::writeRow<double, double, double>(double, double, double);

void CsvWriter::flush() {
    file_.flush();
}

void CsvWriter::close() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

} // namespace wav