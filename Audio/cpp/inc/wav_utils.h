/**
 * @file wav_utils.h
 * @brief Common utilities for WAV file processing (header declarations)
 * Author: Julia Wen (wendigilane@gmail.com)
 * @par Revision History
 * - 11-21-2025 — Initial check-in  
 * - 12-02-2025 — Update with improvement
 */

#ifndef WAV_UTILS_H
#define WAV_UTILS_H

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace wav {

// ============================================================================
// WAV Format Constants
// ============================================================================

extern const double kPI;

extern const uint16_t PCM_16_BPS;
extern const uint16_t PCM_24_BPS;
extern const uint16_t PCM_32_BPS;
extern const uint16_t WAVE_FMT_PCM;
extern const uint16_t WAVE_FMT_FLOAT;

extern const size_t BYTES_PER_SAMPLE_16;
extern const size_t BYTES_PER_SAMPLE_32;

extern const int32_t SIGN_BIT_24;
extern const int32_t SIGN_EXTEND_24;

extern const size_t FLUSH_INTERVAL;

// ============================================================================
// WAV File Structure
// ============================================================================

struct WavFormat {
    uint16_t format;        // PCM=1, Float=3
    uint16_t channels;      // Number of channels
    uint32_t sampleRate;    // Sample rate in Hz
    uint16_t bitsPerSample; // Bits per sample
    uint32_t dataSize;      // Size of audio data in bytes
    std::streampos dataPos; // Position of data chunk in file
    
    size_t frames() const;
    size_t bytesPerSample() const;
    bool isValid() const;
    bool isSupportedFormat() const;
};

// ============================================================================
// Low-level Read Functions
// ============================================================================

uint16_t readU16(std::istream& in);
uint32_t readU32(std::istream& in);
int16_t readI16(const uint8_t* p);
int32_t readI24(const uint8_t* p);

// ============================================================================
// WAV Header Parsing
// ============================================================================

int parseWavHeader(std::istream& f, WavFormat& fmt);

// ============================================================================
// Sample Decoding
// ============================================================================

double decodeSample(const uint8_t* ptr, const WavFormat& fmt);

double decodeSampleMono(const std::vector<uint8_t>& raw, size_t frame, 
                        const WavFormat& fmt);

double decodeSampleChannel(const std::vector<uint8_t>& raw, size_t frame, 
                           size_t channel, const WavFormat& fmt);

// ============================================================================
// Window Functions
// ============================================================================

enum class WindowType { Rectangular, Hann, Hamming, Blackman };

double windowValue(WindowType type, size_t i, size_t N);
WindowType parseWindow(const std::string& w);
std::vector<double> generateWindowCoeffs(WindowType type, size_t N);

// ============================================================================
// CSV Writing Utilities
// ============================================================================

class CsvWriter {
private:
    std::ofstream file_;
    size_t lineCount_;
    
public:
    explicit CsvWriter(const std::string& path);
    ~CsvWriter();
    
    // Disable copy
    CsvWriter(const CsvWriter&) = delete;
    CsvWriter& operator=(const CsvWriter&) = delete;
    
    void writeHeader(const std::string& header);
    void writeLine(const std::string& line);
    
    template<typename T1, typename T2>
    void writeRow(T1 col1, T2 col2);
    
    void flush();
    void close();
};

} // namespace wav

#endif // WAV_UTILS_H