/**
 * @file test_wav_freq_csv_multi_channel.cpp
 * @brief Google Test for wav_freq_csv multi-channel output (1–14 channels)
 * Author:        Julia Wen (wendigilane@gmail.com)
 * Generates WAV files with 1–14 channels, runs the multi-channel
 * wav_freq_csv_channelized program, and verifies:
 *   - CSV per channel has correct number of samples
 *   - Spectrum CSV per channel is non-zero
 *   - Works for all window types (Hann, Hamming, Blackman, Rectangular)
 * 
 * Fully automated: no manual adjustment needed for number of channels.
 * Automatically cleans the output directory before each run.
 */

#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;

// ===== Constants =====
constexpr int kSampleRate = 8000;
constexpr int kNumSamples = 256;
constexpr int kMaxChannels = 14;
constexpr float kSineFreq = 1000.0f;
constexpr int16_t kMaxAmplitude = 10000;
const fs::path kOutDir = "test_output";
const std::string kTestWavFile = "multi_sine.wav";

// ===== Load CSV values (ignore header) =====
std::vector<float> loadCsv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Failed to open CSV file: " + path);
    std::string line;
    std::vector<float> vals;
    bool header = true;
    while (std::getline(f, line)) {
        if (header) { header = false; continue; }
        if (line.empty()) continue;
        auto pos = line.find(',');
        if (pos != std::string::npos) {
            float val = std::stof(line.substr(pos+1));
            vals.push_back(val);
        }
    }
    return vals;
}

// ===== Generate multi-channel sine WAV =====
void generateSineWaveWav(const std::string& path, int sampleRate, int samples,
                         int channels = 1, float freq = kSineFreq) {
    std::vector<int16_t> buf(samples * channels);
    for (int i = 0; i < samples; ++i) {
        for (int ch = 0; ch < channels; ++ch) {
            buf[i * channels + ch] = static_cast<int16_t>(
                kMaxAmplitude * sin(2 * M_PI * freq * (i + ch * 0.1f) / sampleRate));
        }
    }

    std::ofstream f(path, std::ios::binary);
    f.write("RIFF", 4);
    uint32_t chunkSize = 36 + samples * channels * 2;
    f.write(reinterpret_cast<char*>(&chunkSize), 4);
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    uint32_t subChunk1 = 16;
    f.write(reinterpret_cast<char*>(&subChunk1), 4);
    uint16_t audioFormat = 1;
    f.write(reinterpret_cast<char*>(&audioFormat), 2);
    uint16_t numChannels = channels;
    f.write(reinterpret_cast<char*>(&numChannels), 2);
    uint32_t sr = sampleRate;
    f.write(reinterpret_cast<char*>(&sr), 4);
    uint32_t byteRate = sampleRate * channels * 2;
    f.write(reinterpret_cast<char*>(&byteRate), 4);
    uint16_t blockAlign = 2 * channels;
    f.write(reinterpret_cast<char*>(&blockAlign), 2);
    uint16_t bitsPerSample = 16;
    f.write(reinterpret_cast<char*>(&bitsPerSample), 2);

    f.write("data", 4);
    uint32_t dataSize = samples * channels * 2;
    f.write(reinterpret_cast<char*>(&dataSize), 4);
    f.write(reinterpret_cast<char*>(buf.data()), dataSize);
    f.close();
}

// ===== Google Test =====
class WavFreqCsvChannelTest : public ::testing::TestWithParam<std::string> {};

TEST_P(WavFreqCsvChannelTest, MultiChannelSpectrumNotZero) {
    std::string window = GetParam();

    // ===== Clean output directory =====
    if (fs::exists(kOutDir)) fs::remove_all(kOutDir);
    fs::create_directories(kOutDir);

    // ===== Loop over channels 1 → 14 =====
    for (int channels = 1; channels <= kMaxChannels; ++channels) {
        generateSineWaveWav(kTestWavFile, kSampleRate, kNumSamples, channels, kSineFreq);

        std::string cmd = "./wav_freq_csv_channelized " + kTestWavFile + " " + kOutDir.string() + " " + window;
        int ret = std::system(cmd.c_str());
        ASSERT_EQ(ret, 0) << "Failed for window: " << window << " channels: " << channels;

        for (int ch = 0; ch < channels; ++ch) {
            auto csv = kOutDir / ("multi_sine_ch" + std::to_string(ch+1) + ".csv");
            auto spec_csv = kOutDir / ("multi_sine_spectrum_ch" + std::to_string(ch+1) + ".csv");

            auto sample_vals = loadCsv(csv.string());
            ASSERT_EQ(sample_vals.size(), static_cast<size_t>(kNumSamples))
                << "Channel " << ch+1 << " sample count mismatch";

            auto spectrum_vals = loadCsv(spec_csv.string());
            ASSERT_GT(spectrum_vals.size(), 0) << "Channel " << ch+1 << " spectrum empty";

            bool any_nonzero = false;
            for (float v : spectrum_vals) {
                if (v != 0.0f) { any_nonzero = true; break; }
            }
            ASSERT_TRUE(any_nonzero) << "Spectrum all zero for channel " << ch+1;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(AllWindows,
                         WavFreqCsvChannelTest,
                         ::testing::Values("hann","hamming","blackman","rectangular"));
