//=============================================================================
//  FileName:      wav_freq_csv.cpp
//  Author:        Julia Wen (wendigilane@gmail.com)
//  Description:   WAV → CSV + FFT Spectrum
//                 Windowing support: Hann, Hamming, Blackman, Rectangular
//                 C++17: constexpr, auto, structured loops
//  Revision History:
//-----------------------------------------------------------------------------
//  Sep 07, 2025  1.0       Julia Wen    Initial check in
//  Nov 12, 2025  1.1       Julia Wen    Added error_codes.h, error handling, improvement
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

constexpr double kPI = 3.14159265358979323846;

// WAV constants
constexpr uint16_t PCM_16_BPS = 16;
constexpr uint16_t PCM_24_BPS = 24;
constexpr uint16_t WAVE_FMT_PCM = 1;
constexpr uint16_t WAVE_FMT_FLOAT = 3;

constexpr size_t BYTES_PER_SAMPLE_16 = 2;
constexpr size_t BYTES_PER_SAMPLE_24 = 3;
constexpr size_t BYTES_PER_SAMPLE_32 = 4;

// FFT and Bit-reversal permutation
inline void bitReverseInPlace(std::vector<cd>& a) noexcept {
    const size_t n = a.size();
    if (n <= 2) return;

    size_t j = 0;
    for (size_t i = 1; i < n - 1; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

// Faster, branch-reduced FFT using iterative radix-2 Cooley-Tukey
inline int fft(std::vector<cd>& a, bool invert) noexcept {
    const size_t n = a.size();
    if (n == 0) return ERR_FFT_COMPUTE;  // invalid input
    if (n == 1) return SUCCESS;          // nothing to do
    if ((n & (n - 1)) != 0) return ERR_FFT_COMPUTE;  // not a power of two

    bitReverseInPlace(a);

    const double sgn = invert ? -1.0 : 1.0;
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = sgn * 2.0 * kPI / static_cast<double>(len);
        const cd wlen(std::cos(ang), std::sin(ang));

        // Process each block of size 'len'
        for (size_t i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            const size_t half = len >> 1;
            cd* a0 = &a[i];            // local pointer avoids repeated indexing
            cd* a1 = a0 + half;

            for (size_t j = 0; j < half; ++j) {
                const cd u = a0[j];
                const cd v = a1[j] * w;
                a0[j]     = u + v;
                a1[j]     = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        const double invN = 1.0 / static_cast<double>(n);
        for (auto& x : a) x *= invN;
    }

    return SUCCESS;
}

// Window functions
enum class WindowType { Rectangular, Hann, Hamming, Blackman };

constexpr double windowValue(WindowType type, size_t i, size_t N) {
    switch(type) {
        case WindowType::Rectangular: return 1.0;
        case WindowType::Hann:       return 0.5 * (1.0 - std::cos(2.0 * kPI * static_cast<double>(i) / static_cast<double>(N - 1)));
        case WindowType::Hamming:    return 0.54 - 0.46 * std::cos(2.0 * kPI * static_cast<double>(i) / static_cast<double>(N - 1));
        case WindowType::Blackman:   return 0.42 - 0.5 * std::cos(2.0 * kPI * static_cast<double>(i) / static_cast<double>(N - 1)) + 0.08 * std::cos(4.0 * kPI * static_cast<double>(i) / static_cast<double>(N - 1));
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
    uint8_t b[2];
    if (!in.read(reinterpret_cast<char*>(b), static_cast<std::streamsize>(BYTES_PER_SAMPLE_16))) throw ERR_READ_FAILURE;
    return static_cast<uint16_t>(b[0] | (b[1] << 8));
}

static uint32_t readU32(std::istream& in) {
    uint8_t b[4];
    if (!in.read(reinterpret_cast<char*>(b), static_cast<std::streamsize>(BYTES_PER_SAMPLE_32))) throw ERR_READ_FAILURE;
    return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) | (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
}

static int16_t readI16(const uint8_t* p) {
    return static_cast<int16_t>(p[0] | (p[1] << 8));
}

static int32_t readI24(const uint8_t* p) {
    int32_t v = (p[0] | (p[1] << 8) | (p[2] << 16));
    if (v & 0x800000) v |= ~0xFFFFFF;
    return v;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.wav output.csv [window]\n";
        return ERR_INVALID_INPUT;
    }

    const std::string inPath = argv[1];
    const std::string outPath = argv[2];
    const auto window = (argc >= 4) ? parseWindow(argv[3]) : WindowType::Rectangular;

    std::ifstream f(inPath, std::ios::binary);
    if (!f) { std::cerr << "Failed to open input: " << inPath << '\n'; return ERR_FILE_NOT_FOUND; }

    // Read WAV header
    char riff[4]; f.read(riff, 4);
    if (std::string(riff, 4) != "RIFF") { std::cerr << "Not a RIFF file\n"; return ERR_INVALID_HEADER; }
    // chunk size
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
        char id[4];
        if (!f.read(id, 4)) break;
        const auto sz = readU32(f);
        const std::string sid(id, 4);
        if (sid == "fmt ") {
            have_fmt = true;
            fmt = readU16(f);
            channels = readU16(f);
            sampleRate = readU32(f);
            readU32(f); // byte rate
            readU16(f); // block align
            bps = readU16(f);
            if (sz > 16) f.seekg(static_cast<std::streamoff>(sz - 16), std::ios::cur);
        } else if (sid == "data") {
            have_data = true;
            dataPos = f.tellg();
            dataSize = sz;
            f.seekg(static_cast<std::streamoff>(sz + (sz & 1)), std::ios::cur);
        } else {
            f.seekg(static_cast<std::streamoff>(sz + (sz & 1)), std::ios::cur);
        }
    }

    if (!have_fmt || !have_data) { std::cerr << "Missing fmt or data chunk\n"; return ERR_INVALID_HEADER; }
    if (!((fmt == WAVE_FMT_PCM && (bps == PCM_16_BPS || bps == PCM_24_BPS)) || (fmt == WAVE_FMT_FLOAT && bps == 32))) {
        std::cerr << "Only PCM 16/24-bit or Float32 supported\n";
        return ERR_UNSUPPORTED_FORMAT;
    }

    // Read audio data
    f.clear();
    f.seekg(dataPos);
    std::vector<uint8_t> raw;
    try {
        raw.resize(dataSize);
    } catch (...) {
        return ERR_FFT_MEMORY;
    }
    if (!f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(dataSize))) {
        std::cerr << "Failed to read audio data\n";
        return ERR_READ_FAILURE;
    }

    const auto sampleBytes = static_cast<size_t>(bps) / 8;
    const auto frames = static_cast<size_t>(dataSize) / (sampleBytes * static_cast<size_t>(channels));

    // Read samples and apply window
    // Precompute window coefficients
    std::vector<double> windowCoeffs(frames);
    for (size_t i = 0; i < frames; ++i)
        windowCoeffs[i] = windowValue(window, i, frames);

    // Allocate sample vector
    std::vector<double> samples;
    samples.reserve(frames);

    // Open CSV
    std::ofstream csv(outPath);
    if (!csv) { std::cerr << "Failed to open output CSV: " << outPath << '\n'; return ERR_CSV_WRITE_FAILURE; }
    csv << "Index,Sample\n"; // CSV header

    constexpr size_t flushEvery = 1000; // Flush CSV every N lines to ensure buffered data is written periodically
    size_t lineCount = 0;

    // Main loop: decode samples, apply window, write CSV
    for (size_t i = 0; i < frames; ++i) {
        double sample = 0.0;

        // PCM 16-bit
        if (fmt == WAVE_FMT_PCM && bps == PCM_16_BPS) {
            if (channels == 1) sample = readI16(&raw[i * BYTES_PER_SAMPLE_16]);
            else {
                const auto L = readI16(&raw[(i * channels + 0) * BYTES_PER_SAMPLE_16]);
                const auto R = readI16(&raw[(i * channels + 1) * BYTES_PER_SAMPLE_16]);
                sample = (L + R) / 2.0;
            }
        }
        // PCM 24-bit
        else if (fmt == WAVE_FMT_PCM && bps == PCM_24_BPS) {
            if (channels == 1) sample = readI24(&raw[i * BYTES_PER_SAMPLE_24]);
            else {
                const auto L = readI24(&raw[(i * channels + 0) * BYTES_PER_SAMPLE_24]);
                const auto R = readI24(&raw[(i * channels + 1) * BYTES_PER_SAMPLE_24]);
                sample = (L + R) / 2.0;
            }
        }
        // Float32
        else if (fmt == WAVE_FMT_FLOAT && bps == 32) {
            if (channels == 1) sample = *reinterpret_cast<const float*>(&raw[i * BYTES_PER_SAMPLE_32]);
            else {
                const auto L = *reinterpret_cast<const float*>(&raw[(i * channels + 0) * BYTES_PER_SAMPLE_32]);
                const auto R = *reinterpret_cast<const float*>(&raw[(i * channels + 1) * BYTES_PER_SAMPLE_32]);
                sample = (L + R) / 2.0;
            }
        }

        // Apply window
        sample *= windowCoeffs[i];

        // Store sample
        samples.push_back(sample);

        // Write CSV line
        csv << i << "," << sample << '\n';
        if (++lineCount >= flushEvery) { csv.flush(); lineCount = 0; }
    }

    // Final flush
    csv.flush();
    csv.close();

    // FFT
    auto fftN = size_t{1};
    while (fftN < samples.size()) fftN <<= 1;
    std::vector<cd> fa(fftN, 0);
    std::copy(samples.begin(), samples.end(), fa.begin());

    if (auto err = fft(fa, false); err != SUCCESS) return err;

    // Write spectrum CSV
    auto specPath = outPath;
    if (auto dot = specPath.find_last_of('.'); dot != std::string::npos) specPath.insert(dot, "_spectrum");
    else specPath += "_spectrum.csv";

    std::ofstream spec(specPath);
    if (!spec) return ERR_CSV_WRITE_FAILURE;
    spec << "Frequency(Hz),Magnitude\n";
    for (auto i = size_t{0}; i < fftN / 2; ++i) {
        const double freq = static_cast<double>(i) * static_cast<double>(sampleRate) / static_cast<double>(fftN);
        const double mag = std::abs(fa[i]);
        spec << freq << "," << mag << '\n';
    }
    spec.close();

    std::cout << "Wrote " << samples.size() << " samples to " << outPath
              << " and " << specPath << " (" << sampleRate << " Hz)\n";

    return SUCCESS;
}
