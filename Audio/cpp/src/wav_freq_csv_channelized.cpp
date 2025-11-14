//=============================================================================
//  FileName:      wav_freq_csv_channelized.cpp
//  Author:        Julia Wen (wendigilane@gmail.com)
//  Description:   WAV → per-channel CSV + per-channel FFT Spectrum
//                 Channelized (supports up to 14 channels), no downmixing
//                 Thread-safe: per-channel FFT and CSV writing done in parallel
//                 Output: output directory; filenames auto-generated from input
//                         <input_basename>_ch{N}.csv and <input_basename>_ch{N}_spectrum.csv
//                 C++17: constexpr, auto, filesystem, threads
//-----------------------------------------------------------------------------
//  Sep 07, 2025  1.0       Julia Wen    Initial check in
//  Nov 12, 2025  1.1       Julia Wen    Added error_codes.h, error handling, improvement
//  Nov 13, 2025  1.2       Julia Wen    Channelized, threaded, directory output
//=============================================================================

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <thread>
#include <mutex>
#include <filesystem>
#include "../inc/error_codes.h"

using cd = std::complex<double>;

constexpr double kPI = 3.14159265358979323846;
constexpr size_t FLUSH_INTERVAL = 1000;

static std::mutex g_io_mutex;  // thread-safe console output

// WAV constants
constexpr uint16_t PCM_16_BPS = 16;
constexpr uint16_t PCM_24_BPS = 24;
constexpr uint16_t WAVE_FMT_PCM = 1;
constexpr uint16_t WAVE_FMT_FLOAT = 3;

constexpr size_t BYTES_PER_SAMPLE_16 = 2;
constexpr size_t BYTES_PER_SAMPLE_24 = 3;
constexpr size_t BYTES_PER_SAMPLE_32 = 4;

constexpr int MAX_NUM_CHANNELS = 14;

// FFT helpers (bitReverseInPlace and fft) remain unchanged from original file
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
                a0[j] = u + v;
                a1[j] = u - v;
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

// Thread-safe console output helper
void printThreadSafe(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_io_mutex);
    std::cout << msg << std::endl;
}

// Process one channel
void process_channel(size_t ch, size_t channels, size_t frames, const std::vector<uint8_t>& raw,
                     uint16_t fmt, uint16_t bps, WindowType window, uint32_t sampleRate,
                     const std::filesystem::path& outDir, const std::string& inName) {

    std::vector<double> samples(frames);
    std::vector<double> windowCoeffs(frames);
    for (size_t i = 0; i < frames; ++i)
        windowCoeffs[i] = windowValue(window, i, frames);

    const size_t sampleBytes = bps / 8;

    // decode samples
    for (size_t i = 0; i < frames; ++i) {
        double sample = 0.0;
        const uint8_t* ptr = &raw[i * channels * sampleBytes + ch * sampleBytes];
        if (fmt == WAVE_FMT_PCM && bps == PCM_16_BPS) sample = readI16(ptr);
        else if (fmt == WAVE_FMT_PCM && bps == PCM_24_BPS) sample = readI24(ptr);
        else if (fmt == WAVE_FMT_FLOAT && bps == 32) sample = *reinterpret_cast<const float*>(ptr);
         // Apply window
        sample *= windowCoeffs[i];       
        // Store sample
        samples[i] = sample;

    }

    // write CSV
    std::filesystem::path csvPath = outDir / (inName + "_ch" + std::to_string(ch+1) + ".csv");
    std::ofstream csv(csvPath);
    if (!csv) throw ERR_CSV_WRITE_FAILURE;
    csv << "Index,Sample\n";
    size_t lineCount = 0;
    for (size_t i = 0; i < frames; ++i) {
        csv << i << "," << samples[i] << "\n";
        if (++lineCount >= FLUSH_INTERVAL) { csv.flush(); lineCount = 0; }
    }
    csv.flush();
    csv.close();

    // FFT
    size_t fftN = 1;
    while (fftN < samples.size()) fftN <<= 1;
    std::vector<cd> fa(fftN, 0);
    std::copy(samples.begin(), samples.end(), fa.begin());
    if (fft(fa, false) != SUCCESS) throw ERR_FFT_COMPUTE;

    // spectrum CSV
    std::filesystem::path specPath = outDir / (inName + "_spectrum_ch" + std::to_string(ch+1) + ".csv");
    std::ofstream spec(specPath);
    if (!spec) throw ERR_CSV_WRITE_FAILURE;
    spec << "Frequency(Hz),Magnitude\n";
    for (size_t i = 0; i < fftN / 2; ++i) {
        double freq = i * static_cast<double>(sampleRate) / fftN;
        double mag = std::abs(fa[i]);
        spec << freq << "," << mag << "\n";
    }
    spec.close();

    printThreadSafe("Channel " + std::to_string(ch+1) + " done: " +
                    csvPath.string() + ", " + specPath.string());
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.wav output_dir [window]\n";
        return ERR_INVALID_INPUT;
    }

    std::filesystem::path inPath = argv[1];
    std::filesystem::path outDir = argv[2];
    std::string inName = inPath.stem().string();
    WindowType window = (argc >= 4) ? parseWindow(argv[3]) : WindowType::Rectangular;

    std::ifstream f(inPath, std::ios::binary);
    if (!f) { std::cerr << "Failed to open input: " << inPath << '\n'; return ERR_FILE_NOT_FOUND; }

    // read WAV header (same as original)
    char riff[4]; f.read(riff,4);
    if (std::string(riff,4)!="RIFF") return ERR_INVALID_HEADER;
    readU32(f);
    char wave[4]; f.read(wave,4);
    if (std::string(wave,4)!="WAVE") return ERR_INVALID_HEADER;

    bool have_fmt=false, have_data=false;
    uint16_t fmt=0, channels=0, bps=0;
    uint32_t sampleRate=0;
    std::streampos dataPos{};
    uint32_t dataSize=0;

    while(f && !(have_fmt && have_data)) {
        char id[4]; if(!f.read(id,4)) break;
        uint32_t sz = readU32(f);
        std::string sid(id,4);
        if(sid=="fmt ") {
            have_fmt=true;
            fmt = readU16(f);
            channels = readU16(f);
            sampleRate = readU32(f);
            readU32(f);
            readU16(f);
            bps = readU16(f);
            if(sz>16) f.seekg(sz-16,std::ios::cur);
        } else if(sid=="data") {
            have_data=true;
            dataPos = f.tellg();
            dataSize = sz;
            f.seekg(sz + (sz &1), std::ios::cur);
        } else f.seekg(sz + (sz&1), std::ios::cur);
    }

    if(!have_fmt || !have_data) return ERR_INVALID_HEADER;
    if(!((fmt==WAVE_FMT_PCM && (bps==PCM_16_BPS||bps==PCM_24_BPS))||(fmt==WAVE_FMT_FLOAT && bps==32)))
        return ERR_UNSUPPORTED_FORMAT;

    f.clear(); f.seekg(dataPos);
    std::vector<uint8_t> raw(dataSize);
    if(!f.read(reinterpret_cast<char*>(raw.data()),dataSize)) return ERR_READ_FAILURE;

    // launch threads per channel
    std::vector<std::thread> threads;
    for(size_t ch=0; ch<channels && ch<MAX_NUM_CHANNELS; ++ch) {
        threads.emplace_back(process_channel, ch, channels, dataSize/(bps/8)/channels,
                             std::cref(raw), fmt, bps, window, sampleRate,
                             std::cref(outDir), inName);
    }
    for(auto& t: threads) t.join();

    return SUCCESS;
}
