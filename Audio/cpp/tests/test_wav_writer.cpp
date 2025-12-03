/**
 * @file test_wav_writer.cpp
 * @brief Unit tests (gtest) for WavWriter class
 * @author Julia Wen (wendigilane@gmail.com)
 * @date 12-02-2025
 */

#include <gtest/gtest.h>
#include "../inc/wav_writer.h"
#include <fstream>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>

// Helper class to read and validate WAV files
class WavReader {
public:
    explicit WavReader(const std::string& filename) {
        file_.open(filename, std::ios::binary);
        if (!file_.is_open()) {
            throw std::runtime_error("Failed to open WAV file for reading");
        }
        readHeader();
    }

    int getSampleRate() const { return sampleRate_; }
    int getNumChannels() const { return numChannels_; }
    int getBitDepth() const { return bitDepth_; }
    uint32_t getDataSize() const { return dataSize_; }
    bool isFloat() const { return audioFormat_ == 3; }

    // Read one sample (returns as int32 for all formats)
    int32_t readSampleAsInt32() {
        if (bitDepth_ == 16) {
            int16_t val;
            file_.read(reinterpret_cast<char*>(&val), 2);
            return static_cast<int32_t>(val);
        } else if (bitDepth_ == 24) {
            uint8_t bytes[3];
            file_.read(reinterpret_cast<char*>(bytes), 3);
            int32_t val = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16);
            // Sign extend from 24-bit to 32-bit
            if (val & 0x800000) {
                val |= 0xFF000000;
            }
            return val;
        } else if (bitDepth_ == 32 && !isFloat()) {
            int32_t val;
            file_.read(reinterpret_cast<char*>(&val), 4);
            return val;
        }
        throw std::runtime_error("Cannot read as int32 from float format");
    }

    // Read one sample as float
    float readSampleAsFloat() {
        if (isFloat()) {
            // Read 4 bytes as little-endian float
            uint8_t bytes[4];
            file_.read(reinterpret_cast<char*>(bytes), 4);
            
            // Reinterpret as float (little-endian)
            uint32_t intVal = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
            float val;
            std::memcpy(&val, &intVal, sizeof(float));
            return val;
        } else if (bitDepth_ == 16) {
            int16_t val;
            file_.read(reinterpret_cast<char*>(&val), 2);
            return static_cast<float>(val) / 32767.0f;
        } else if (bitDepth_ == 24) {
            int32_t val = readSampleAsInt32();
            return static_cast<float>(val) / 8388607.0f;
        }
        throw std::runtime_error("Unsupported format for float reading");
    }

    bool eof() const { return file_.eof(); }

private:
    void readHeader() {
        char buffer[44];
        file_.read(buffer, 44);

        // Verify "RIFF"
        if (std::strncmp(buffer, "RIFF", 4) != 0) {
            throw std::runtime_error("Not a valid WAV file (missing RIFF)");
        }

        // Verify "WAVE"
        if (std::strncmp(buffer + 8, "WAVE", 4) != 0) {
            throw std::runtime_error("Not a valid WAV file (missing WAVE)");
        }

        // Parse fmt chunk
        audioFormat_ = *reinterpret_cast<uint16_t*>(buffer + 20);
        numChannels_ = *reinterpret_cast<uint16_t*>(buffer + 22);
        sampleRate_ = *reinterpret_cast<uint32_t*>(buffer + 24);
        bitDepth_ = *reinterpret_cast<uint16_t*>(buffer + 34);
        dataSize_ = *reinterpret_cast<uint32_t*>(buffer + 40);
    }

    std::ifstream file_;
    uint16_t audioFormat_;
    int sampleRate_;
    int numChannels_;
    int bitDepth_;
    uint32_t dataSize_;
};

// ============================================================================
// Google Test Fixture
// ============================================================================

class WavWriterTest : public ::testing::Test {
protected:
    const std::string testFile_ = "test_output.wav";

    void TearDown() override {
        std::remove(testFile_.c_str());
    }

    bool fileExists(const std::string& filename) const {
        std::ifstream f(filename);
        return f.good();
    }
};

// ============================================================================
// Construction and Basic Properties Tests
// ============================================================================

TEST_F(WavWriterTest, ConstructValid16Bit) {
    EXPECT_NO_THROW({
        WavWriter writer(testFile_, 44100, 2, 16);
        EXPECT_TRUE(writer.isOpen());
        EXPECT_EQ(writer.getSampleRate(), 44100);
        EXPECT_EQ(writer.getNumChannels(), 2);
        EXPECT_EQ(writer.getBitDepth(), 16);
    });
}

TEST_F(WavWriterTest, ConstructValid24Bit) {
    EXPECT_NO_THROW({
        WavWriter writer(testFile_, 48000, 1, 24);
        EXPECT_EQ(writer.getBitDepth(), 24);
    });
}

TEST_F(WavWriterTest, ConstructValid32Bit) {
    EXPECT_NO_THROW({
        WavWriter writer(testFile_, 96000, 2, 32);
        EXPECT_EQ(writer.getBitDepth(), 32);
    });
}

TEST_F(WavWriterTest, ConstructInvalidBitDepth) {
    EXPECT_THROW({
        WavWriter writer(testFile_, 44100, 2, 8);
    }, std::runtime_error);
}

TEST_F(WavWriterTest, ConstructInvalidChannels) {
    EXPECT_THROW({
        WavWriter writer(testFile_, 44100, 0, 16);
    }, std::runtime_error);
}

TEST_F(WavWriterTest, ConstructInvalidSampleRate) {
    EXPECT_THROW({
        WavWriter writer(testFile_, -1, 2, 16);
    }, std::runtime_error);
}

TEST_F(WavWriterTest, ConstructInvalidFilename) {
    EXPECT_THROW({
        WavWriter writer("/invalid/path/that/does/not/exist/file.wav", 44100, 2, 16);
    }, std::runtime_error);
}

// ============================================================================
// Header Validation Tests
// ============================================================================

TEST_F(WavWriterTest, HeaderFormat16Bit) {
    {
        WavWriter writer(testFile_, 44100, 2, 16);
    } // Destructor finalizes

    WavReader reader(testFile_);
    EXPECT_EQ(reader.getSampleRate(), 44100);
    EXPECT_EQ(reader.getNumChannels(), 2);
    EXPECT_EQ(reader.getBitDepth(), 16);
    EXPECT_FALSE(reader.isFloat());
}

TEST_F(WavWriterTest, HeaderFormat32BitFloat) {
    {
        WavWriter writer(testFile_, 48000, 1, 32);
    }

    WavReader reader(testFile_);
    EXPECT_EQ(reader.getBitDepth(), 32);
    EXPECT_TRUE(reader.isFloat());
}

// ============================================================================
// 16-bit Sample Writing Tests
// ============================================================================

TEST_F(WavWriterTest, Write16BitSamples) {
    {
        WavWriter writer(testFile_, 44100, 1, 16);
        writer.writeSample(int16_t(1000));
        writer.writeSample(int16_t(-1000));
        writer.writeSample(int16_t(0));
    }

    WavReader reader(testFile_);
    EXPECT_EQ(reader.readSampleAsInt32(), 1000);
    EXPECT_EQ(reader.readSampleAsInt32(), -1000);
    EXPECT_EQ(reader.readSampleAsInt32(), 0);
}

TEST_F(WavWriterTest, Write16BitMaxMin) {
    {
        WavWriter writer(testFile_, 44100, 1, 16);
        writer.writeSample(int16_t(32767));   // Max
        writer.writeSample(int16_t(-32768));  // Min
    }

    WavReader reader(testFile_);
    EXPECT_EQ(reader.readSampleAsInt32(), 32767);
    EXPECT_EQ(reader.readSampleAsInt32(), -32768);
}

// ============================================================================
// 24-bit Sample Writing Tests
// ============================================================================

TEST_F(WavWriterTest, Write16BitTo24BitConversion) {
    {
        WavWriter writer(testFile_, 44100, 1, 24);
        writer.writeSample(int16_t(32767));  // Max 16-bit
        writer.writeSample(int16_t(-32768)); // Min 16-bit
        writer.writeSample(int16_t(100));
    }

    WavReader reader(testFile_);
    // 16-bit to 24-bit should shift left by 8 bits
    EXPECT_EQ(reader.readSampleAsInt32(), 8388352);   // 32767 << 8
    EXPECT_EQ(reader.readSampleAsInt32(), -8388608);  // -32768 << 8
    EXPECT_EQ(reader.readSampleAsInt32(), 25600);     // 100 << 8
}

TEST_F(WavWriterTest, Write24BitDirect) {
    {
        WavWriter writer(testFile_, 44100, 1, 24);
        writer.writeSample(int32_t(8388607));   // Max 24-bit
        writer.writeSample(int32_t(-8388608));  // Min 24-bit
        writer.writeSample(int32_t(1000000));
    }

    WavReader reader(testFile_);
    EXPECT_EQ(reader.readSampleAsInt32(), 8388607);
    EXPECT_EQ(reader.readSampleAsInt32(), -8388608);
    EXPECT_EQ(reader.readSampleAsInt32(), 1000000);
}

TEST_F(WavWriterTest, Write24BitClamping) {
    {
        WavWriter writer(testFile_, 44100, 1, 24);
        // Write values outside 24-bit range
        writer.writeSample(int32_t(10000000));   // Above max
        writer.writeSample(int32_t(-10000000));  // Below min
    }

    WavReader reader(testFile_);
    // Should be clamped to 24-bit range
    EXPECT_EQ(reader.readSampleAsInt32(), 8388607);
    EXPECT_EQ(reader.readSampleAsInt32(), -8388608);
}

// ============================================================================
// Float Sample Writing Tests
// ============================================================================

TEST_F(WavWriterTest, WriteFloatTo32BitFloat) {
    {
        WavWriter writer(testFile_, 44100, 1, 32);
        writer.writeSample(0.5f);
        writer.writeSample(-0.5f);
        writer.writeSample(0.0f);
        writer.writeSample(1.0f);
        writer.writeSample(-1.0f);
    }

    WavReader reader(testFile_);
    EXPECT_NEAR(reader.readSampleAsFloat(), 0.5f, 0.0001f);
    EXPECT_NEAR(reader.readSampleAsFloat(), -0.5f, 0.0001f);
    EXPECT_NEAR(reader.readSampleAsFloat(), 0.0f, 0.0001f);
    EXPECT_NEAR(reader.readSampleAsFloat(), 1.0f, 0.0001f);
    EXPECT_NEAR(reader.readSampleAsFloat(), -1.0f, 0.0001f);
}

TEST_F(WavWriterTest, WriteFloatTo16Bit) {
    {
        WavWriter writer(testFile_, 44100, 1, 16);
        writer.writeSample(1.0f);    // Should map to 32767
        writer.writeSample(-1.0f);   // Should map to -32768
        writer.writeSample(0.5f);    // Should map to ~16383
    }

    WavReader reader(testFile_);
    EXPECT_EQ(reader.readSampleAsInt32(), 32767);
    EXPECT_EQ(reader.readSampleAsInt32(), -32768);
    EXPECT_NEAR(reader.readSampleAsInt32(), 16383, 1);
}

TEST_F(WavWriterTest, WriteFloatTo24Bit) {
    {
        WavWriter writer(testFile_, 44100, 1, 24);
        writer.writeSample(1.0f);   // Should map to 8388607
        writer.writeSample(-1.0f);  // Should map to -8388608
    }

    WavReader reader(testFile_);
    EXPECT_EQ(reader.readSampleAsInt32(), 8388607);
    EXPECT_EQ(reader.readSampleAsInt32(), -8388608);
}

TEST_F(WavWriterTest, FloatClamping) {
    {
        WavWriter writer(testFile_, 44100, 1, 32);
        writer.writeSample(2.0f);   // Above range
        writer.writeSample(-2.0f);  // Below range
    }

    WavReader reader(testFile_);
    EXPECT_NEAR(reader.readSampleAsFloat(), 1.0f, 0.0001f);
    EXPECT_NEAR(reader.readSampleAsFloat(), -1.0f, 0.0001f);
}

// ============================================================================
// Int32 Sample Writing Tests
// ============================================================================

TEST_F(WavWriterTest, WriteInt32To32BitFloat) {
    {
        WavWriter writer(testFile_, 44100, 1, 32);
        writer.writeSample(int32_t(2147483647));  // INT32_MAX
        writer.writeSample(int32_t(-2147483648)); // INT32_MIN
        writer.writeSample(int32_t(0));
    }

    WavReader reader(testFile_);
    float val1 = reader.readSampleAsFloat();
    float val2 = reader.readSampleAsFloat();
    float val3 = reader.readSampleAsFloat();
    
    EXPECT_NEAR(val1, 1.0f, 0.001f);
    EXPECT_FLOAT_EQ(val2, -1.0f);
    EXPECT_FLOAT_EQ(val3, 0.0f);
}

TEST_F(WavWriterTest, WriteInt32To16Bit) {
    {
        WavWriter writer(testFile_, 44100, 1, 16);
        // Int32 to 16-bit uses upper 16 bits
        writer.writeSample(int32_t(0x7FFF0000));  // Max in upper bits
        writer.writeSample(int32_t(0x80000000));  // Min in upper bits
    }

    WavReader reader(testFile_);
    EXPECT_EQ(reader.readSampleAsInt32(), 32767);
    EXPECT_EQ(reader.readSampleAsInt32(), -32768);
}

// ============================================================================
// Bulk Write Tests
// ============================================================================

TEST_F(WavWriterTest, WriteSamplesArray) {
    std::vector<float> samples = {0.1f, 0.2f, 0.3f, -0.1f, -0.2f};
    
    {
        WavWriter writer(testFile_, 44100, 1, 32);
        writer.writeSamples(samples.data(), samples.size());
    }

    WavReader reader(testFile_);
    for (float expected : samples) {
        EXPECT_FLOAT_EQ(reader.readSampleAsFloat(), expected);
    }
}

TEST_F(WavWriterTest, WriteSamplesInt32Array) {
    std::vector<int32_t> samples = {1000, 2000, -1000, -2000, 0};
    
    {
        WavWriter writer(testFile_, 44100, 1, 24);
        writer.writeSamples(samples.data(), samples.size());
    }

    WavReader reader(testFile_);
    for (int32_t expected : samples) {
        EXPECT_EQ(reader.readSampleAsInt32(), expected);
    }
}

TEST_F(WavWriterTest, WriteNullPointer) {
    WavWriter writer(testFile_, 44100, 1, 16);
    EXPECT_THROW(writer.writeSamples(static_cast<float*>(nullptr), 10), std::invalid_argument);
    EXPECT_THROW(writer.writeSamples(static_cast<int32_t*>(nullptr), 10), std::invalid_argument);
}

// ============================================================================
// Multi-Channel Frame Tests
// ============================================================================

TEST_F(WavWriterTest, WriteStereoFrames) {
    {
        WavWriter writer(testFile_, 44100, 2, 16);
        
        float frame1[2] = {0.5f, -0.5f};
        float frame2[2] = {0.25f, -0.25f};
        
        writer.writeFrame(frame1);
        writer.writeFrame(frame2);
    }

    WavReader reader(testFile_);
    EXPECT_EQ(reader.getNumChannels(), 2);
    
    // Read interleaved: L1, R1, L2, R2
    EXPECT_NEAR(reader.readSampleAsInt32(), 16383, 1);
    EXPECT_NEAR(reader.readSampleAsInt32(), -16384, 1);
    EXPECT_NEAR(reader.readSampleAsInt32(), 8191, 1);
    EXPECT_NEAR(reader.readSampleAsInt32(), -8192, 1);
}

TEST_F(WavWriterTest, WriteMultiChannelInt32Frames) {
    {
        WavWriter writer(testFile_, 44100, 3, 24);
        
        int32_t frame[3] = {1000, 2000, 3000};
        writer.writeFrame(frame);
    }

    WavReader reader(testFile_);
    EXPECT_EQ(reader.getNumChannels(), 3);
    EXPECT_EQ(reader.readSampleAsInt32(), 1000);
    EXPECT_EQ(reader.readSampleAsInt32(), 2000);
    EXPECT_EQ(reader.readSampleAsInt32(), 3000);
}

TEST_F(WavWriterTest, WriteFrameNullPointer) {
    WavWriter writer(testFile_, 44100, 2, 16);
    EXPECT_THROW(writer.writeFrame(static_cast<float*>(nullptr)), std::invalid_argument);
    EXPECT_THROW(writer.writeFrame(static_cast<int32_t*>(nullptr)), std::invalid_argument);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(WavWriterTest, MoveConstructor) {
    WavWriter writer1(testFile_, 44100, 2, 16);
    writer1.writeSample(int16_t(1000));
    
    WavWriter writer2(std::move(writer1));
    
    EXPECT_TRUE(writer2.isOpen());
    EXPECT_FALSE(writer1.isOpen());
    EXPECT_EQ(writer2.getSampleRate(), 44100);
    
    // Should be able to continue writing with writer2
    EXPECT_NO_THROW(writer2.writeSample(int16_t(2000)));
}

TEST_F(WavWriterTest, MoveAssignment) {
    WavWriter writer1(testFile_, 44100, 2, 16);
    writer1.writeSample(int16_t(1000));
    
    WavWriter writer2("temp.wav", 48000, 1, 24);
    writer2 = std::move(writer1);
    
    EXPECT_TRUE(writer2.isOpen());
    EXPECT_EQ(writer2.getSampleRate(), 44100);
    EXPECT_EQ(writer2.getBitDepth(), 16);
    
    std::remove("temp.wav");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(WavWriterTest, WriteAfterDestruction) {
    WavWriter* writer = new WavWriter(testFile_, 44100, 1, 16);
    delete writer;
    
    // File should exist and be valid
    EXPECT_TRUE(fileExists(testFile_));
    WavReader reader(testFile_);
    EXPECT_EQ(reader.getBitDepth(), 16);
}

TEST_F(WavWriterTest, FlushMethod) {
    WavWriter writer(testFile_, 44100, 1, 16);
    writer.writeSample(int16_t(1000));
    
    EXPECT_NO_THROW(writer.flush());
}

// ============================================================================
// Data Size Tests
// ============================================================================

TEST_F(WavWriterTest, DataSizeCalculation16Bit) {
    {
        WavWriter writer(testFile_, 44100, 1, 16);
        // Write 100 samples
        for (int i = 0; i < 100; ++i) {
            writer.writeSample(int16_t(i));
        }
    }

    WavReader reader(testFile_);
    // 100 samples × 2 bytes = 200 bytes
    EXPECT_EQ(reader.getDataSize(), 200);
}

TEST_F(WavWriterTest, DataSizeCalculation24Bit) {
    {
        WavWriter writer(testFile_, 44100, 1, 24);
        for (int i = 0; i < 100; ++i) {
            writer.writeSample(int32_t(i * 1000));
        }
    }

    WavReader reader(testFile_);
    // 100 samples × 3 bytes = 300 bytes
    EXPECT_EQ(reader.getDataSize(), 300);
}

TEST_F(WavWriterTest, DataSizeCalculation32BitFloat) {
    {
        WavWriter writer(testFile_, 44100, 2, 32);
        for (int i = 0; i < 50; ++i) {
            float frame[2] = {0.1f * i, -0.1f * i};
            writer.writeFrame(frame);
        }
    }

    WavReader reader(testFile_);
    // 50 frames × 2 channels × 4 bytes = 400 bytes
    EXPECT_EQ(reader.getDataSize(), 400);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(WavWriterTest, WriteZeroSamples) {
    {
        WavWriter writer(testFile_, 44100, 1, 16);
        // Don't write anything
    }

    WavReader reader(testFile_);
    EXPECT_EQ(reader.getDataSize(), 0);
}

TEST_F(WavWriterTest, WriteOneSample) {
    {
        WavWriter writer(testFile_, 44100, 1, 16);
        writer.writeSample(int16_t(12345));
    }

    WavReader reader(testFile_);
    EXPECT_EQ(reader.getDataSize(), 2);
    EXPECT_EQ(reader.readSampleAsInt32(), 12345);
}

TEST_F(WavWriterTest, HighSampleRate) {
    EXPECT_NO_THROW({
        WavWriter writer(testFile_, 192000, 8, 24);
        EXPECT_EQ(writer.getSampleRate(), 192000);
        EXPECT_EQ(writer.getNumChannels(), 8);
    });
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}