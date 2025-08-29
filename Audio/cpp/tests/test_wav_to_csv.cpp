#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

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

TEST(WavToCsvTest, GenerateCsv) {
    const char* wav = "sine.wav";
    const char* csv = "sine.csv";
    int sampleRate = 8000;
    int samples = 128;
    std::vector<int16_t> buf(samples);
    for (int i=0;i<samples;i++) {
        buf[i] = (int16_t)(10000 * sin(2*M_PI*440*i/sampleRate));
    }
    std::ofstream f(wav, std::ios::binary);
    f.write("RIFF",4); uint32_t sz = 36+samples*2; f.write((char*)&sz,4); f.write("WAVE",4);
    f.write("fmt ",4); uint32_t sub=16; f.write((char*)&sub,4); uint16_t fmt=1; f.write((char*)&fmt,2);
    uint16_t ch=1; f.write((char*)&ch,2); uint32_t sr=sampleRate; f.write((char*)&sr,4);
    uint32_t br=sr*2; f.write((char*)&br,4); uint16_t ba=2; f.write((char*)&ba,2);
    uint16_t bps=16; f.write((char*)&bps,2);
    f.write("data",4); uint32_t dsz=samples*2; f.write((char*)&dsz,4);
    f.write((char*)buf.data(),dsz); f.close();

    // Call wav_to_csv binary from project root
    int ret = std::system(("./wav_to_csv " + std::string(wav) + " " + csv).c_str());
    ASSERT_EQ(ret, 0);

    auto samples_csv = loadCsv(csv);
    ASSERT_EQ(samples_csv.size(), samples);
    bool any_nonzero = false;
    for (float v: samples_csv) { if (v!=0.0f) { any_nonzero = true; break; } }
    ASSERT_TRUE(any_nonzero);
}
