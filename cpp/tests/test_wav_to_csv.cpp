// cpp/tests/test_wav_to_csv.cpp
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <sstream>

// Helper: write 16-bit PCM WAV
void writeWav16(const std::string& path, const std::vector<int16_t>& samples) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Failed to open file");

    uint32_t sampleRate = 16000;
    uint16_t channels = 1;
    uint16_t bitsPerSample = 16;
    uint32_t byteRate = sampleRate * channels * bitsPerSample / 8;
    uint16_t blockAlign = channels * bitsPerSample / 8;
    uint32_t dataSize = samples.size() * blockAlign;

    f.write("RIFF", 4);
    uint32_t chunkSize = 36 + dataSize;
    f.write(reinterpret_cast<const char*>(&chunkSize), 4);
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    uint32_t subchunk1Size = 16;
    uint16_t audioFormat = 1; // PCM
    f.write(reinterpret_cast<const char*>(&subchunk1Size), 4);
    f.write(reinterpret_cast<const char*>(&audioFormat), 2);
    f.write(reinterpret_cast<const char*>(&channels), 2);
    f.write(reinterpret_cast<const char*>(&sampleRate), 4);
    f.write(reinterpret_cast<const char*>(&byteRate), 4);
    f.write(reinterpret_cast<const char*>(&blockAlign), 2);
    f.write(reinterpret_cast<const char*>(&bitsPerSample), 2);

    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&dataSize), 4);
    f.write(reinterpret_cast<const char*>(samples.data()), dataSize);
}

// Helper: write 24-bit PCM WAV
void writeWav24(const std::string& path, const std::vector<int32_t>& samples) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Failed to open file");

    uint32_t sampleRate = 16000;
    uint16_t channels = 1;
    uint16_t bitsPerSample = 24;
    uint32_t byteRate = sampleRate * channels * bitsPerSample / 8;
    uint16_t blockAlign = channels * bitsPerSample / 8;
    uint32_t dataSize = samples.size() * blockAlign;

    f.write("RIFF", 4);
    uint32_t chunkSize = 36 + dataSize;
    f.write(reinterpret_cast<const char*>(&chunkSize), 4);
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    uint32_t subchunk1Size = 16;
    uint16_t audioFormat = 1; // PCM
    f.write(reinterpret_cast<const char*>(&subchunk1Size), 4);
    f.write(reinterpret_cast<const char*>(&audioFormat), 2);
    f.write(reinterpret_cast<const char*>(&channels), 2);
    f.write(reinterpret_cast<const char*>(&sampleRate), 4);
    f.write(reinterpret_cast<const char*>(&byteRate), 4);
    f.write(reinterpret_cast<const char*>(&blockAlign), 2);
    f.write(reinterpret_cast<const char*>(&bitsPerSample), 2);

    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&dataSize), 4);
    for (auto s : samples) {
        char bytes[3];
        bytes[0] = s & 0xFF;
        bytes[1] = (s >> 8) & 0xFF;
        bytes[2] = (s >> 16) & 0xFF;
        f.write(bytes, 3);
    }
}

// Helper: write float32 IEEE WAV
void writeWavFloat32(const std::string& path, const std::vector<float>& samples) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Failed to open file");

    uint32_t sampleRate = 16000;
    uint16_t channels = 1;
    uint16_t bitsPerSample = 32;
    uint32_t byteRate = sampleRate * channels * bitsPerSample / 8;
    uint16_t blockAlign = channels * bitsPerSample / 8;
    uint32_t dataSize = samples.size() * blockAlign;

    f.write("RIFF", 4);
    uint32_t chunkSize = 36 + dataSize;
    f.write(reinterpret_cast<const char*>(&chunkSize), 4);
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    uint32_t subchunk1Size = 16;
    uint16_t audioFormat = 3; // IEEE float
    f.write(reinterpret_cast<const char*>(&subchunk1Size), 4);
    f.write(reinterpret_cast<const char*>(&audioFormat), 2);
    f.write(reinterpret_cast<const char*>(&channels), 2);
    f.write(reinterpret_cast<const char*>(&sampleRate), 4);
    f.write(reinterpret_cast<const char*>(&byteRate), 4);
    f.write(reinterpret_cast<const char*>(&blockAlign), 2);
    f.write(reinterpret_cast<const char*>(&bitsPerSample), 2);

    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&dataSize), 4);
    f.write(reinterpret_cast<const char*>(samples.data()), dataSize);
}

// Helper: load CSV
std::vector<float> loadCsv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Failed to open CSV file");

    std::string line;
    std::vector<float> samples;
    bool header = true;
    while (std::getline(f, line)) {
        if (header) { header = false; continue; }
        if (line.empty()) continue;
        auto pos = line.find(',');
        if (pos != std::string::npos) {
            float val = std::stof(line.substr(pos + 1));
            samples.push_back(val);
        }
    }
    return samples;
}

// Actual test
class WavToCsvTest : public ::testing::Test {
protected:
    std::vector<int16_t> pcm16 {0, 1000, -1000, 0, 32767, -32768, 0};
    std::vector<int32_t> pcm24 {0, 100000, -100000, 0, 500000, -500000, 0};
    std::vector<float>   float32 {0.0f, 0.5f, -0.5f, 0.0f, 1.0f, -1.0f, 0.0f};
};

TEST_F(WavToCsvTest, Pcm16ToCsv) {
    std::string wav = "test16.wav";
    std::string csv = "test16.csv";
    writeWav16(wav, pcm16);

    int ret = std::system(("./cpp/audio/wav_to_csv " + wav + " " + csv).c_str());
    ASSERT_EQ(ret, 0);

    auto got = loadCsv(csv);
    ASSERT_EQ(got.size(), pcm16.size());
    for (size_t i = 0; i < got.size(); i++) {
        if (pcm16[i] == 0)
            EXPECT_FLOAT_EQ(got[i], 0.0f);
        else
            EXPECT_NE(got[i], 0.0f);
    }
}

TEST_F(WavToCsvTest, Pcm24ToCsv) {
    std::string wav = "test24.wav";
    std::string csv = "test24.csv";
    writeWav24(wav, pcm24);

    int ret = std::system(("./cpp/audio/wav_to_csv " + wav + " " + csv).c_str());
    ASSERT_EQ(ret, 0);

    auto got = loadCsv(csv);
    ASSERT_EQ(got.size(), pcm24.size());
    for (size_t i = 0; i < got.size(); i++) {
        if (pcm24[i] == 0)
            EXPECT_FLOAT_EQ(got[i], 0.0f);
        else
            EXPECT_NE(got[i], 0.0f);
    }
}

TEST_F(WavToCsvTest, Float32ToCsv) {
    std::string wav = "test32.wav";
    std::string csv = "test32.csv";
    writeWavFloat32(wav, float32);

    int ret = std::system(("./cpp/audio/wav_to_csv " + wav + " " + csv).c_str());
    ASSERT_EQ(ret, 0);

    auto got = loadCsv(csv);
    ASSERT_EQ(got.size(), float32.size());
    for (size_t i = 0; i < got.size(); i++) {
        if (float32[i] == 0.0f)
            EXPECT_FLOAT_EQ(got[i], 0.0f);
        else
            EXPECT_NE(got[i], 0.0f);
    }
}

