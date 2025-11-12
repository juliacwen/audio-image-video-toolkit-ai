//=============================================================================
//  FileName:      wav_freq_csv.cpp
//  Author:        Julia Wen
//  Date:          September 7, 2025
//  Description:   WAV → CSV + FFT Spectrum
//                 Windowing support: Hann, Hamming, Blackman, Rectangular
//                 C++17: constexpr, auto, structured loops
//=============================================================================

//=============================================================================
//  Revision History:
//-----------------------------------------------------------------------------
//  Sep 07, 2025  1.0       Julia Wen    Original creation
//  Nov 12, 2025  1.1       Julia Wen    Added error_codes.h and proper error handling
//=============================================================================

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include "../inc/error_codes.h"   // Added for error handling

using cd = std::complex<double>;

// Use literal for PI to allow constexpr in C++17
constexpr double kPI = 3.14159265358979323846;

// WAV constants
constexpr uint16_t PCM_16_BPS = 16;
constexpr uint16_t PCM_24_BPS = 24;
constexpr uint16_t WAVE_FMT_PCM = 1;
constexpr uint16_t WAVE_FMT_FLOAT = 3;

// FFT and Bit-reversal permutation
void bitReverse(std::vector<cd>& a) {
    const auto n = a.size();
    size_t j = 0;
    for (size_t i = 1; i < n; i++) {
        size_t bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

int fft(std::vector<cd>& a, bool invert) {
    try {
        const auto n = a.size();
        bitReverse(a);
        for (size_t len = 2; len <= n; len <<= 1) {
            const double ang = 2 * kPI / len * (invert ? -1 : 1);
            const cd wlen(cos(ang), sin(ang));
            for (size_t i = 0; i < n; i += len) {
                cd w(1);
                for (size_t j = 0; j < len / 2; ++j) {
                    auto u = a[i + j];
                    auto v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (invert) for (auto& x : a) x /= n;
        return SUCCESS;
    } catch (...) {
        return ERR_FFT_COMPUTE;
    }
}

// Window functions
enum class WindowType { Rectangular, Hann, Hamming, Blackman };

constexpr double windowValue(WindowType type, size_t i, size_t N) {
    switch(type) {
        case WindowType::Rectangular: return 1.0;
        case WindowType::Hann:       return 0.5 * (1 - std::cos(2 * kPI * i / (N - 1)));
        case WindowType::Hamming:    return 0.54 - 0.46 * std::cos(2 * kPI * i / (N - 1));
        case WindowType::Blackman:   return 0.42 - 0.5 * std::cos(2 * kPI * i / (N - 1)) + 0.08 * std::cos(4 * kPI * i / (N - 1));
    }
    return 1.0;
}

// Parse window type from string
WindowType parseWindow(const std::string& w) {
    std::string s = w;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "hann") return WindowType::Hann;
    if (s == "hamming") return WindowType::Hamming;
    if (s == "blackman") return WindowType::Blackman;
    return WindowType::Rectangular;
}

// WAV reading helpers
static uint16_t readU16(std::istream& in) {
    uint8_t b[2]; in.read(reinterpret_cast<char*>(b), 2);
    return b[0] | (b[1] << 8);
}

static uint32_t readU32(std::istream& in) {
    uint8_t b[4]; in.read(reinterpret_cast<char*>(b), 4);
    return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
}

static int16_t readI16(const uint8_t* p) {
    return int16_t(p[0] | (p[1] << 8));
}

static int32_t readI24(const uint8_t* p) {
    int32_t v = (p[0] | (p[1] << 8) | (p[2] << 16));
    if (v & 0x800000) v |= ~0xFFFFFF;
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) return ERR_UNKNOWN;

    const std::string inPath = argv[1];
    const std::string outPath = argv[2];
    WindowType window = (argc >= 4) ? parseWindow(argv[3]) : WindowType::Rectangular;

    std::ifstream f(inPath, std::ios::binary);
    if (!f) { std::cerr << "Failed to open input: " << inPath << "\n"; return ERR_FILE_NOT_FOUND; }

    // Read WAV header
    char riff[4]; f.read(riff, 4);
    if (std::string(riff, 4) != "RIFF") { std::cerr << "Not a RIFF file\n"; return ERR_INVALID_HEADER; }
    readU32(f);
    char wave[4]; f.read(wave, 4);
    if (std::string(wave, 4) != "WAVE") { std::cerr << "Not a WAVE file\n"; return ERR_INVALID_HEADER; }

    bool have_fmt = false, have_data = false;
    uint16_t fmt = 0, channels = 0, bps = 0;
    uint32_t sampleRate = 0;
    std::streampos dataPos{};
    uint32_t dataSize = 0;

    // Parse chunks
    while (f && !(have_fmt && have_data)) {
        char id[4]; if (!f.read(id, 4)) break;
        uint32_t sz = readU32(f);
        std::string sid(id, 4);
        if (sid == "fmt ") {
            have_fmt = true;
            fmt = readU16(f);
            channels = readU16(f);
            sampleRate = readU32(f);
            readU32(f); readU16(f);
            bps = readU16(f);
            if (sz > 16) f.seekg(sz - 16, std::ios::cur);
        } else if (sid == "data") {
            have_data = true;
            dataPos = f.tellg();
            dataSize = sz;
            f.seekg(sz + (sz & 1), std::ios::cur);
        } else {
            f.seekg(sz + (sz & 1), std::ios::cur);
        }
    }

    if (!have_fmt || !have_data) { std::cerr << "Missing fmt or data chunk\n"; return ERR_INVALID_HEADER; }
    if (!((fmt == WAVE_FMT_PCM && (bps == PCM_16_BPS || bps == PCM_24_BPS)) || (fmt == WAVE_FMT_FLOAT && bps == 32))) {
        std::cerr << "Only PCM 16/24-bit or Float32 supported\n"; 
        return ERR_UNSUPPORTED_FORMAT;
    }

    // Read audio data
    f.clear(); f.seekg(dataPos);
    std::vector<uint8_t> raw(dataSize);
    if (!f.read(reinterpret_cast<char*>(raw.data()), dataSize)) {
        std::cerr << "Failed to read audio data\n";
        return ERR_READ_FAILURE;
    }

    const size_t sampleBytes = bps / 8;
    const size_t frames = dataSize / (sampleBytes * channels);

    std::ofstream csv(outPath);
    if (!csv) { std::cerr << "Failed to open output CSV: " << outPath << "\n"; return ERR_CSV_WRITE_FAILURE; }
    csv << "Index,Sample\n";

    std::vector<double> samples;
    samples.reserve(frames);

    for (size_t i = 0; i < frames; ++i) {
        double sample = 0.0;

        if (fmt == WAVE_FMT_PCM) {
            if (bps == PCM_16_BPS) {
                if (channels == 1) sample = readI16(&raw[i * 2]);
                else {
                    int32_t L = readI16(&raw[(i * channels + 0) * 2]);
                    int32_t R = readI16(&raw[(i * channels + 1) * 2]);
                    sample = (L + R) / 2.0;
                }
            } else if (bps == PCM_24_BPS) {
                if (channels == 1) sample = readI24(&raw[i * 3]);
                else {
                    int32_t L = readI24(&raw[(i * channels + 0) * 3]);
                    int32_t R = readI24(&raw[(i * channels + 1) * 3]);
                    sample = (L + R) / 2.0;
                }
            }
        } else if (fmt == WAVE_FMT_FLOAT && bps == 32) {
            if (channels == 1) sample = *reinterpret_cast<const float*>(&raw[i * 4]);
            else {
                float L = *reinterpret_cast<const float*>(&raw[(i * channels + 0) * 4]);
                float R = *reinterpret_cast<const float*>(&raw[(i * channels + 1) * 4]);
                sample = (L + R) / 2.0;
            }
        }

        // Apply window
        sample *= windowValue(window, i, frames);

        csv << i << "," << sample << "\n";
        samples.push_back(sample);
    }
    csv.close();

    // FFT
    size_t fftN = 1; while (fftN < samples.size()) fftN <<= 1;
    std::vector<cd> fa(fftN, 0);
    std::copy(samples.begin(), samples.end(), fa.begin());

    if (int err = fft(fa, false); err != SUCCESS) return err;

    // Write spectrum CSV
    std::string specPath = outPath;
    const auto dot = specPath.find_last_of('.');
    if (dot != std::string::npos) specPath.insert(dot, "_spectrum");
    else specPath += "_spectrum.csv";

    std::ofstream spec(specPath);
    if (!spec) return ERR_CSV_WRITE_FAILURE;
    spec << "Frequency(Hz),Magnitude\n";
    for (size_t i = 0; i < fftN / 2; ++i) {
        double freq = static_cast<double>(i) * sampleRate / fftN;
        double mag = std::abs(fa[i]);
        spec << freq << "," << mag << "\n";
    }
    spec.close();

    std::cout << "Wrote " << samples.size() << " samples to " << outPath
              << " and " << specPath << " (" << sampleRate << " Hz)\n";

    return SUCCESS;
}
