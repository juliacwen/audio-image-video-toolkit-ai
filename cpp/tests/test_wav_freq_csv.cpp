#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

std::vector<float> loadCsv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Failed to open CSV file");
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

void generateSineWaveWav(const std::string& path, int sampleRate, int samples, float freq=1000.0f) {
    std::vector<int16_t> buf(samples);
    for (int i=0;i<samples;i++) {
        buf[i] = static_cast<int16_t>(10000 * sin(2*M_PI*freq*i/sampleRate));
    }
    std::ofstream f(path, std::ios::binary);
    f.write("RIFF",4);
    uint32_t chunkSize = 36 + samples*2;
    f.write(reinterpret_cast<char*>(&chunkSize),4);
    f.write("WAVE",4);
    f.write("fmt ",4);
    uint32_t subChunk1 = 16;
    f.write(reinterpret_cast<char*>(&subChunk1),4);
    uint16_t audioFormat = 1;
    f.write(reinterpret_cast<char*>(&audioFormat),2);
    uint16_t numChannels = 1;
    f.write(reinterpret_cast<char*>(&numChannels),2);
    uint32_t sr = sampleRate;
    f.write(reinterpret_cast<char*>(&sr),4);
    uint32_t byteRate = sampleRate*2;
    f.write(reinterpret_cast<char*>(&byteRate),4);
    uint16_t blockAlign = 2;
    f.write(reinterpret_cast<char*>(&blockAlign),2);
    uint16_t bitsPerSample = 16;
    f.write(reinterpret_cast<char*>(&bitsPerSample),2);
    f.write("data",4);
    uint32_t dataSize = samples*2;
    f.write(reinterpret_cast<char*>(&dataSize),4);
    f.write(reinterpret_cast<char*>(buf.data()),dataSize);
    f.close();
}

class WavFreqCsvWindowTest : public ::testing::TestWithParam<std::string> {};

TEST_P(WavFreqCsvWindowTest, SpectrumNotZero) {
    std::string window = GetParam();
    const std::string wav = "sine.wav";
    const std::string csv = "sine.csv";
    const std::string spec_csv = "sine_spectrum.csv";
    int sampleRate = 8000;
    int samples = 256;

    generateSineWaveWav(wav, sampleRate, samples, 1000.0f);

    std::string cmd = "./wav_freq_csv " + wav + " " + csv + " 0 " + window;
    int ret = std::system(cmd.c_str());
    ASSERT_EQ(ret, 0) << "Failed for window: " << window;

    auto sample_vals = loadCsv(csv);
    ASSERT_EQ(sample_vals.size(), static_cast<size_t>(samples));

    auto spectrum_vals = loadCsv(spec_csv);
    ASSERT_GT(spectrum_vals.size(), 0);

    bool any_nonzero = false;
    for (float v: spectrum_vals) {
        if (v != 0.0f) { any_nonzero = true; break; }
    }
    ASSERT_TRUE(any_nonzero);
}

INSTANTIATE_TEST_SUITE_P(AllWindows,
                         WavFreqCsvWindowTest,
                         ::testing::Values("hann","hamming","blackman","rectangular"));

