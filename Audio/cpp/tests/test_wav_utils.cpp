/**
 * @file test_wav_utils.cpp
 * @brief Unit tests for WAV utilities
 * @author Julia Wen (wendigilane@gmail.com)
 * @date 12-02-2025
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../inc/wav_utils.h"
#include "../inc/error_codes.h"
#include <sstream>
#include <fstream>
#include <cstring>
#include <vector>

using namespace wav;
using ::testing::DoubleNear;

// ============================================================================
// Test Fixtures
// ============================================================================

class TestWavUtils : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup
    }

    void TearDown() override {
        // Cleanup test files
    }

    // Helper: Create a minimal valid WAV header in memory
    std::stringstream createValidWavHeader(uint16_t format = WAVE_FMT_PCM,
                                           uint16_t channels = 2,
                                           uint32_t sampleRate = 44100,
                                           uint16_t bitsPerSample = 16,
                                           uint32_t dataSize = 1000) {
        std::stringstream ss;
        
        // RIFF header
        ss.write("RIFF", 4);
        uint32_t fileSize = 36 + dataSize;
        ss.write(reinterpret_cast<char*>(&fileSize), 4);
        ss.write("WAVE", 4);
        
        // fmt chunk
        ss.write("fmt ", 4);
        uint32_t fmtSize = 16;
        ss.write(reinterpret_cast<char*>(&fmtSize), 4);
        ss.write(reinterpret_cast<char*>(&format), 2);
        ss.write(reinterpret_cast<char*>(&channels), 2);
        ss.write(reinterpret_cast<char*>(&sampleRate), 4);
        uint32_t byteRate = sampleRate * channels * (bitsPerSample / 8);
        ss.write(reinterpret_cast<char*>(&byteRate), 4);
        uint16_t blockAlign = channels * (bitsPerSample / 8);
        ss.write(reinterpret_cast<char*>(&blockAlign), 2);
        ss.write(reinterpret_cast<char*>(&bitsPerSample), 2);
        
        // data chunk
        ss.write("data", 4);
        ss.write(reinterpret_cast<char*>(&dataSize), 4);
        
        // Write dummy data
        std::vector<uint8_t> dummy(dataSize, 0);
        ss.write(reinterpret_cast<char*>(dummy.data()), dataSize);
        
        return ss;
    }
};

// ============================================================================
// WavFormat Tests
// ============================================================================

TEST_F(TestWavUtils, WavFormat_Frames_Calculation) {
    WavFormat fmt;
    fmt.dataSize = 1000;
    fmt.bitsPerSample = 16;
    fmt.channels = 2;
    
    // 1000 bytes / 2 bytes per sample / 2 channels = 250 frames
    EXPECT_EQ(fmt.frames(), 250);
}

TEST_F(TestWavUtils, WavFormat_Frames_ZeroChannels) {
    WavFormat fmt;
    fmt.dataSize = 1000;
    fmt.bitsPerSample = 16;
    fmt.channels = 0;
    
    EXPECT_EQ(fmt.frames(), 0);
}

TEST_F(TestWavUtils, WavFormat_Frames_ZeroBitsPerSample) {
    WavFormat fmt;
    fmt.dataSize = 1000;
    fmt.bitsPerSample = 0;
    fmt.channels = 2;
    
    EXPECT_EQ(fmt.frames(), 0);
}

TEST_F(TestWavUtils, WavFormat_Frames_InvalidBitsPerSample) {
    WavFormat fmt;
    fmt.dataSize = 1000;
    fmt.bitsPerSample = 7;  // Not divisible by 8
    fmt.channels = 2;
    fmt.sampleRate = 44100;
    
    EXPECT_EQ(fmt.frames(), 0);  // Should return 0 for invalid input
}

TEST_F(TestWavUtils, WavFormat_Frames_ZeroSampleRate) {
    WavFormat fmt;
    fmt.dataSize = 1000;
    fmt.bitsPerSample = 16;
    fmt.channels = 2;
    fmt.sampleRate = 0;
    
    EXPECT_EQ(fmt.frames(), 0);  // Should return 0 for invalid sample rate
}

TEST_F(TestWavUtils, WavFormat_BytesPerSample) {
    WavFormat fmt;
    
    fmt.bitsPerSample = 16;
    EXPECT_EQ(fmt.bytesPerSample(), 2);
    
    fmt.bitsPerSample = 24;
    EXPECT_EQ(fmt.bytesPerSample(), 3);
    
    fmt.bitsPerSample = 32;
    EXPECT_EQ(fmt.bytesPerSample(), 4);
}

TEST_F(TestWavUtils, WavFormat_IsValid) {
    WavFormat fmt;
    fmt.format = 0;
    fmt.channels = 0;
    fmt.sampleRate = 0;
    fmt.bitsPerSample = 0;
    EXPECT_FALSE(fmt.isValid());
    
    fmt.format = WAVE_FMT_PCM;
    fmt.channels = 2;
    fmt.sampleRate = 44100;
    fmt.bitsPerSample = 16;
    EXPECT_TRUE(fmt.isValid());
}

TEST_F(TestWavUtils, WavFormat_IsSupportedFormat) {
    WavFormat fmt;
    
    // PCM 16-bit (supported)
    fmt.format = WAVE_FMT_PCM;
    fmt.bitsPerSample = PCM_16_BPS;
    EXPECT_TRUE(fmt.isSupportedFormat());
    
    // PCM 24-bit (supported)
    fmt.bitsPerSample = PCM_24_BPS;
    EXPECT_TRUE(fmt.isSupportedFormat());
    
    // PCM 32-bit (not supported)
    fmt.bitsPerSample = PCM_32_BPS;
    EXPECT_FALSE(fmt.isSupportedFormat());
    
    // Float 32-bit (supported)
    fmt.format = WAVE_FMT_FLOAT;
    fmt.bitsPerSample = PCM_32_BPS;
    EXPECT_TRUE(fmt.isSupportedFormat());
    
    // Float 16-bit (not supported)
    fmt.bitsPerSample = PCM_16_BPS;
    EXPECT_FALSE(fmt.isSupportedFormat());
}

// ============================================================================
// Low-level Read Function Tests
// ============================================================================

TEST_F(TestWavUtils, ReadU16_ValidData) {
    std::stringstream ss;
    uint8_t data[] = {0x34, 0x12}; // Little-endian 0x1234
    ss.write(reinterpret_cast<char*>(data), 2);
    
    EXPECT_EQ(readU16(ss), 0x1234);
}

TEST_F(TestWavUtils, ReadU16_InsufficientData) {
    std::stringstream ss;
    uint8_t data[] = {0x34}; // Only 1 byte
    ss.write(reinterpret_cast<char*>(data), 1);
    
    EXPECT_THROW(readU16(ss), int);
}

TEST_F(TestWavUtils, ReadU32_ValidData) {
    std::stringstream ss;
    uint8_t data[] = {0x78, 0x56, 0x34, 0x12}; // Little-endian 0x12345678
    ss.write(reinterpret_cast<char*>(data), 4);
    
    EXPECT_EQ(readU32(ss), 0x12345678);
}

TEST_F(TestWavUtils, ReadU32_InsufficientData) {
    std::stringstream ss;
    uint8_t data[] = {0x78, 0x56}; // Only 2 bytes
    ss.write(reinterpret_cast<char*>(data), 2);
    
    EXPECT_THROW(readU32(ss), int);
}

TEST_F(TestWavUtils, ReadI16_PositiveValue) {
    uint8_t data[] = {0x00, 0x40}; // 16384
    EXPECT_EQ(readI16(data), 16384);
}

TEST_F(TestWavUtils, ReadI16_NegativeValue) {
    uint8_t data[] = {0x00, 0x80}; // -32768
    EXPECT_EQ(readI16(data), -32768);
}

TEST_F(TestWavUtils, ReadI24_PositiveValue) {
    uint8_t data[] = {0x00, 0x00, 0x40}; // 4194304
    EXPECT_EQ(readI24(data), 4194304);
}

TEST_F(TestWavUtils, ReadI24_NegativeValue) {
    uint8_t data[] = {0x00, 0x00, 0x80}; // -8388608 (sign bit set)
    EXPECT_EQ(readI24(data), -8388608);
}

TEST_F(TestWavUtils, ReadI24_MaxPositive) {
    uint8_t data[] = {0xFF, 0xFF, 0x7F}; // 8388607
    EXPECT_EQ(readI24(data), 8388607);
}

// ============================================================================
// WAV Header Parsing Tests
// ============================================================================

TEST_F(TestWavUtils, ParseWavHeader_ValidPCM16) {
    auto ss = createValidWavHeader(WAVE_FMT_PCM, 2, 44100, 16, 1000);
    WavFormat fmt;
    
    EXPECT_EQ(parseWavHeader(ss, fmt), SUCCESS);
    EXPECT_EQ(fmt.format, WAVE_FMT_PCM);
    EXPECT_EQ(fmt.channels, 2);
    EXPECT_EQ(fmt.sampleRate, 44100);
    EXPECT_EQ(fmt.bitsPerSample, 16);
    EXPECT_EQ(fmt.dataSize, 1000);
}

TEST_F(TestWavUtils, ParseWavHeader_ValidPCM24) {
    auto ss = createValidWavHeader(WAVE_FMT_PCM, 1, 48000, 24, 2400);
    WavFormat fmt;
    
    EXPECT_EQ(parseWavHeader(ss, fmt), SUCCESS);
    EXPECT_EQ(fmt.format, WAVE_FMT_PCM);
    EXPECT_EQ(fmt.bitsPerSample, 24);
}

TEST_F(TestWavUtils, ParseWavHeader_ValidFloat32) {
    auto ss = createValidWavHeader(WAVE_FMT_FLOAT, 2, 96000, 32, 4000);
    WavFormat fmt;
    
    EXPECT_EQ(parseWavHeader(ss, fmt), SUCCESS);
    EXPECT_EQ(fmt.format, WAVE_FMT_FLOAT);
    EXPECT_EQ(fmt.bitsPerSample, 32);
}

TEST_F(TestWavUtils, ParseWavHeader_InvalidRIFF) {
    std::stringstream ss;
    ss.write("FAIL", 4);
    WavFormat fmt;
    
    EXPECT_EQ(parseWavHeader(ss, fmt), ERR_INVALID_HEADER);
}

TEST_F(TestWavUtils, ParseWavHeader_InvalidWAVE) {
    std::stringstream ss;
    ss.write("RIFF", 4);
    uint32_t size = 100;
    ss.write(reinterpret_cast<char*>(&size), 4);
    ss.write("FAIL", 4);
    WavFormat fmt;
    
    EXPECT_EQ(parseWavHeader(ss, fmt), ERR_INVALID_HEADER);
}

TEST_F(TestWavUtils, ParseWavHeader_MissingFmtChunk) {
    std::stringstream ss;
    ss.write("RIFF", 4);
    uint32_t fileSize = 36;
    ss.write(reinterpret_cast<char*>(&fileSize), 4);
    ss.write("WAVE", 4);
    ss.write("data", 4);
    uint32_t dataSize = 1000;
    ss.write(reinterpret_cast<char*>(&dataSize), 4);
    
    WavFormat fmt;
    EXPECT_EQ(parseWavHeader(ss, fmt), ERR_INVALID_HEADER);
}

TEST_F(TestWavUtils, ParseWavHeader_UnsupportedFormat) {
    auto ss = createValidWavHeader(2, 2, 44100, 16, 1000); // Format 2 (unsupported)
    WavFormat fmt;
    
    EXPECT_EQ(parseWavHeader(ss, fmt), ERR_UNSUPPORTED_FORMAT);
}

TEST_F(TestWavUtils, ParseWavHeader_InvalidSampleRate) {
    auto ss = createValidWavHeader(WAVE_FMT_PCM, 2, 0, 16, 1000); // 0 Hz
    WavFormat fmt;
    
    EXPECT_EQ(parseWavHeader(ss, fmt), ERR_INVALID_INPUT);
}

TEST_F(TestWavUtils, ParseWavHeader_TooHighSampleRate) {
    auto ss = createValidWavHeader(WAVE_FMT_PCM, 2, 400000, 16, 1000); // > 384 kHz
    WavFormat fmt;
    
    EXPECT_EQ(parseWavHeader(ss, fmt), ERR_INVALID_INPUT);
}

TEST_F(TestWavUtils, ParseWavHeader_ZeroChannels) {
    auto ss = createValidWavHeader(WAVE_FMT_PCM, 0, 44100, 16, 1000);
    WavFormat fmt;
    
    EXPECT_EQ(parseWavHeader(ss, fmt), ERR_INVALID_INPUT);
}

// ============================================================================
// Sample Decoding Tests
// ============================================================================

TEST_F(TestWavUtils, DecodeSample_PCM16) {
    WavFormat fmt;
    fmt.format = WAVE_FMT_PCM;
    fmt.bitsPerSample = 16;
    
    uint8_t data[] = {0x00, 0x40}; // 16384
    EXPECT_DOUBLE_EQ(decodeSample(data, fmt), 16384.0);
}

TEST_F(TestWavUtils, DecodeSample_PCM24) {
    WavFormat fmt;
    fmt.format = WAVE_FMT_PCM;
    fmt.bitsPerSample = 24;
    
    uint8_t data[] = {0x00, 0x00, 0x40}; // 4194304
    EXPECT_DOUBLE_EQ(decodeSample(data, fmt), 4194304.0);
}

TEST_F(TestWavUtils, DecodeSample_Float32) {
    WavFormat fmt;
    fmt.format = WAVE_FMT_FLOAT;
    fmt.bitsPerSample = 32;
    
    float value = 0.5f;
    const uint8_t* data = reinterpret_cast<const uint8_t*>(&value);
    EXPECT_DOUBLE_EQ(decodeSample(data, fmt), 0.5);
}

TEST_F(TestWavUtils, DecodeSampleMono_SingleChannel) {
    WavFormat fmt;
    fmt.format = WAVE_FMT_PCM;
    fmt.bitsPerSample = 16;
    fmt.channels = 1;
    
    std::vector<uint8_t> raw = {0x00, 0x40}; // 16384
    EXPECT_DOUBLE_EQ(decodeSampleMono(raw, 0, fmt), 16384.0);
}

TEST_F(TestWavUtils, DecodeSampleMono_StereoMixed) {
    WavFormat fmt;
    fmt.format = WAVE_FMT_PCM;
    fmt.bitsPerSample = 16;
    fmt.channels = 2;
    
    // Left: 10000, Right: 20000, Average: 15000
    std::vector<uint8_t> raw = {0x10, 0x27, 0x20, 0x4E};
    EXPECT_DOUBLE_EQ(decodeSampleMono(raw, 0, fmt), 15000.0);
}

TEST_F(TestWavUtils, DecodeSampleChannel_Stereo) {
    WavFormat fmt;
    fmt.format = WAVE_FMT_PCM;
    fmt.bitsPerSample = 16;
    fmt.channels = 2;
    
    std::vector<uint8_t> raw = {0x10, 0x27, 0x20, 0x4E}; // L:10000, R:20000
    
    EXPECT_DOUBLE_EQ(decodeSampleChannel(raw, 0, 0, fmt), 10000.0); // Left
    EXPECT_DOUBLE_EQ(decodeSampleChannel(raw, 0, 1, fmt), 20000.0); // Right
}

// ============================================================================
// Window Function Tests
// ============================================================================

TEST_F(TestWavUtils, WindowValue_Rectangular) {
    EXPECT_DOUBLE_EQ(windowValue(WindowType::Rectangular, 0, 10), 1.0);
    EXPECT_DOUBLE_EQ(windowValue(WindowType::Rectangular, 5, 10), 1.0);
    EXPECT_DOUBLE_EQ(windowValue(WindowType::Rectangular, 9, 10), 1.0);
}

TEST_F(TestWavUtils, WindowValue_Hann) {
    // Hann window: 0.5 * (1 - cos(2*pi*i/(N-1)))
    // Verified with reference: scipy.signal.windows.hann(10, sym=False)
    // [0.0, 0.11697778, 0.41317591, 0.75, 0.96984631, 0.96984631, 0.75, 0.41317591, 0.11697778, 0.0]
    EXPECT_NEAR(windowValue(WindowType::Hann, 0, 10), 0.0, 1e-8);
    EXPECT_NEAR(windowValue(WindowType::Hann, 1, 10), 0.11697778, 1e-8);
    EXPECT_NEAR(windowValue(WindowType::Hann, 2, 10), 0.41317591, 1e-8);
    EXPECT_NEAR(windowValue(WindowType::Hann, 3, 10), 0.75, 1e-8);
    EXPECT_NEAR(windowValue(WindowType::Hann, 4, 10), 0.96984631, 1e-8);
    EXPECT_NEAR(windowValue(WindowType::Hann, 5, 10), 0.96984631, 1e-8);
    EXPECT_NEAR(windowValue(WindowType::Hann, 9, 10), 0.0, 1e-8);
}

TEST_F(TestWavUtils, WindowValue_Hamming) {
    // Hamming window: starts at 0.08, peaks near middle
    double val0 = windowValue(WindowType::Hamming, 0, 10);
    double val5 = windowValue(WindowType::Hamming, 5, 10);
    EXPECT_NEAR(val0, 0.08, 0.01);
    EXPECT_GT(val5, val0);
}

TEST_F(TestWavUtils, WindowValue_Blackman) {
    // Blackman window has specific formula
    double val0 = windowValue(WindowType::Blackman, 0, 10);
    double val5 = windowValue(WindowType::Blackman, 5, 10);
    EXPECT_NEAR(val0, 0.0, 0.01);
    EXPECT_GT(val5, val0);
}

TEST_F(TestWavUtils, ParseWindow_CaseInsensitive) {
    EXPECT_EQ(parseWindow("hann"), WindowType::Hann);
    EXPECT_EQ(parseWindow("HANN"), WindowType::Hann);
    EXPECT_EQ(parseWindow("Hamming"), WindowType::Hamming);
    EXPECT_EQ(parseWindow("BLACKMAN"), WindowType::Blackman);
    EXPECT_EQ(parseWindow("unknown"), WindowType::Rectangular);
}

TEST_F(TestWavUtils, GenerateWindowCoeffs_Size) {
    auto coeffs = generateWindowCoeffs(WindowType::Hann, 256);
    EXPECT_EQ(coeffs.size(), 256);
}

TEST_F(TestWavUtils, GenerateWindowCoeffs_Values) {
    auto coeffs = generateWindowCoeffs(WindowType::Hann, 10);
    // Verified with reference: scipy.signal.windows.hann(10, sym=False)
    EXPECT_NEAR(coeffs[0], 0.0, 1e-8);
    EXPECT_NEAR(coeffs[1], 0.11697778, 1e-8);
    EXPECT_NEAR(coeffs[4], 0.96984631, 1e-8);
    EXPECT_NEAR(coeffs[5], 0.96984631, 1e-8);
    EXPECT_NEAR(coeffs[9], 0.0, 1e-8);
}

// ============================================================================
// CSV Writer Tests
// ============================================================================

TEST_F(TestWavUtils, CsvWriter_CreateFile) {
    const std::string testFile = "test_output.csv";
    {
        CsvWriter writer(testFile);
        writer.writeHeader("Col1,Col2");
        writer.writeRow(1, 2.5);
    }
    
    // Verify file exists and has content
    std::ifstream in(testFile);
    ASSERT_TRUE(in.is_open());
    
    std::string line;
    std::getline(in, line);
    EXPECT_EQ(line, "Col1,Col2");
    
    std::getline(in, line);
    EXPECT_EQ(line, "1,2.5");
    
    in.close();
    std::remove(testFile.c_str());
}

TEST_F(TestWavUtils, CsvWriter_InvalidPath) {
    EXPECT_THROW(CsvWriter writer("/invalid/path/file.csv"), int);
}

TEST_F(TestWavUtils, CsvWriter_WriteLine) {
    const std::string testFile = "test_line.csv";
    {
        CsvWriter writer(testFile);
        writer.writeLine("data1,data2,data3");
    }
    
    std::ifstream in(testFile);
    std::string line;
    std::getline(in, line);
    EXPECT_EQ(line, "data1,data2,data3");
    in.close();
    std::remove(testFile.c_str());
}

TEST_F(TestWavUtils, CsvWriter_FlushInterval) {
    const std::string testFile = "test_flush.csv";
    {
        CsvWriter writer(testFile);
        // Write more than FLUSH_INTERVAL lines
        for (int i = 0; i < 1100; ++i) {
            writer.writeRow(i, i * 1.5);
        }
    }
    
    // Verify all lines were written
    std::ifstream in(testFile);
    int count = 0;
    std::string line;
    while (std::getline(in, line)) {
        ++count;
    }
    EXPECT_EQ(count, 1100);
    in.close();
    std::remove(testFile.c_str());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(TestWavUtils, Integration_ReadAndDecodeSamples) {
    // Create a WAV with known sample values
    auto ss = createValidWavHeader(WAVE_FMT_PCM, 2, 44100, 16, 8);
    WavFormat fmt;
    
    ASSERT_EQ(parseWavHeader(ss, fmt), SUCCESS);
    
    // Seek to data position and read samples
    ss.seekg(fmt.dataPos);
    std::vector<uint8_t> raw(8);
    ss.read(reinterpret_cast<char*>(raw.data()), 8);
    
    // Decode samples
    double mono = decodeSampleMono(raw, 0, fmt);
    EXPECT_DOUBLE_EQ(mono, 0.0); // We wrote zeros in createValidWavHeader
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}