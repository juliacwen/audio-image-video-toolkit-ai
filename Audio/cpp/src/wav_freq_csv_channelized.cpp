/**
 * @file wav_freq_csv_channelized.cpp
 * @brief Multi-channel WAV audio processor with windowing and overlapping FFT spectrum analysis
 * @author Julia Wen (wendigilane@gmail.com)
 * This program reads a multi-channel WAV audio file (PCM 16/24-bit or Float 32-bit),
 * applies an optional window function, and outputs per-channel:
 *   1. A CSV file containing the time-domain samples (windowed)
 *   2. A CSV file containing the frequency spectrum (FFT magnitude)
 *   3. Optional: Overlapping FFT spectrogram for each channel
 * 
 * Each channel is processed independently in parallel using separate threads for
 * improved performance on multi-core systems.
 * 
 * Supported formats:
 *   - PCM 16-bit signed integer
 *   - PCM 24-bit signed integer
 *   - IEEE Float 32-bit
 * 
 * Supported window functions:
 *   - Rectangular (none)
 *   - Hann
 *   - Hamming
 *   - Blackman
 * 
 * Channel limit: Processes up to 14 channels (MAX_NUM_CHANNELS)
 * 
 * Usage: wav_freq_csv_channelized input.wav output_dir [options]
 *   Options:
 *     --window <type>       Window function: rectangular, hann, hamming, blackman (default: rectangular)
 *     --overlap <percent>   Overlap percentage: 0-99 (default: 0, no overlap)
 *     --fft-size <samples>  FFT window size in samples (default: entire file)
 *     --hop-size <samples>  Hop size in samples (overrides overlap percentage)
 * 
 * Output files (per channel):
 *   - output_dir/filename_chN.csv: Index,Sample (windowed time-domain data for first window)
 *   - output_dir/filename_spectrum_chN.csv: Frequency(Hz),Magnitude (averaged FFT spectrum)
 *   - output_dir/filename_spectrogram_chN.csv: Time(s),Frequency(Hz),Magnitude (if overlapping)
 * 
 * @par Revision History
 * - 11-21-2025 — Initial check-in
 * - 12-04-2025 — Added overlapping FFT support
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <future>
#include <mutex>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include "../inc/error_codes.h"
#include "../inc/wav_utils.h"
#include "../inc/fft_utils.h"

constexpr int MAX_NUM_CHANNELS = 14;

static std::mutex g_io_mutex;

struct ProcessingOptions {
    wav::WindowType window = wav::WindowType::Rectangular;
    size_t fftSize = 0;          // 0 means use entire file
    double overlapPercent = 0.0;  // 0-99
    size_t hopSize = 0;           // If set, overrides overlapPercent
    
    size_t getHopSize(size_t windowSize) const {
        if (hopSize > 0) {
            return hopSize;
        }
        return static_cast<size_t>(windowSize * (1.0 - overlapPercent / 100.0));
    }
};

void printThreadSafe(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_io_mutex);
    std::cout << msg << std::endl;
}

void printErrorThreadSafe(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_io_mutex);
    std::cerr << msg << std::endl;
}

ProcessingOptions parseCommandLine(int argc, char** argv) {
    ProcessingOptions opts;
    
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--window" && i + 1 < argc) {
            opts.window = wav::parseWindow(argv[++i]);
        }
        else if (arg == "--overlap" && i + 1 < argc) {
            opts.overlapPercent = std::stod(argv[++i]);
            if (opts.overlapPercent < 0 || opts.overlapPercent >= 100) {
                std::cerr << "Warning: Overlap must be 0-99%. Using 0%.\n";
                opts.overlapPercent = 0;
            }
        }
        else if (arg == "--fft-size" && i + 1 < argc) {
            opts.fftSize = std::stoull(argv[++i]);
        }
        else if (arg == "--hop-size" && i + 1 < argc) {
            opts.hopSize = std::stoull(argv[++i]);
        }
        else if (arg.find("--") != 0) {
            // Legacy: window name without flag
            opts.window = wav::parseWindow(arg);
        }
    }
    
    return opts;
}

void processChannel(size_t ch, const std::vector<uint8_t>& raw, 
                   const wav::WavFormat& fmt, const ProcessingOptions& opts,
                   const std::filesystem::path& outDir, const std::string& inName) {
    
    size_t totalFrames = fmt.frames();
    
    // Decode all samples for this channel first
    std::vector<double> allSamples(totalFrames);
    for (size_t i = 0; i < totalFrames; ++i) {
        allSamples[i] = wav::decodeSampleChannel(raw, i, ch, fmt);
    }
    
    // Determine FFT window size
    size_t windowSize = (opts.fftSize > 0) ? opts.fftSize : totalFrames;
    if (windowSize > totalFrames) {
        windowSize = totalFrames;
    }
    
    size_t hopSize = opts.getHopSize(windowSize);
    
    // Generate window coefficients
    auto windowCoeffs = wav::generateWindowCoeffs(opts.window, windowSize);
    
    // Calculate number of windows
    size_t numWindows = 1;
    if (hopSize > 0 && totalFrames >= windowSize) {
        numWindows = (totalFrames - windowSize) / hopSize + 1;
    }
    
    // Storage for averaged spectrum
    std::vector<double> avgMagnitudes;
    
    // Storage for spectrogram (if overlapping)
    std::vector<std::vector<double>> spectrogram;
    std::vector<double> windowTimes;
    
    // Process each window
    for (size_t w = 0; w < numWindows; ++w) {
        size_t startIdx = w * hopSize;
        if (startIdx + windowSize > totalFrames) break;
        
        // Extract and window samples
        std::vector<double> windowedSamples(windowSize);
        for (size_t i = 0; i < windowSize; ++i) {
            windowedSamples[i] = allSamples[startIdx + i] * windowCoeffs[i];
        }
        
        // Save first window to time-domain CSV
        if (w == 0) {
            std::filesystem::path csvPath = outDir / (inName + "_ch" + std::to_string(ch+1) + ".csv");
            wav::CsvWriter csv(csvPath.string());
            csv.writeHeader("Index,Sample");
            for (size_t i = 0; i < windowSize; ++i) {
                csv.writeRow(i, windowedSamples[i]);
            }
            csv.close();
        }
        
        // Compute FFT
        auto fftData = fft::prepareForFFT(windowedSamples);
        if (fft::compute(fftData, false) != SUCCESS) {
            throw ERR_FFT_COMPUTE;
        }
        
        auto magnitudes = fft::getMagnitudeSpectrum(fftData);
        
        // Initialize or accumulate
        if (avgMagnitudes.empty()) {
            avgMagnitudes = magnitudes;
        } else {
            for (size_t i = 0; i < magnitudes.size(); ++i) {
                avgMagnitudes[i] += magnitudes[i];
            }
        }
        
        // Store for spectrogram
        if (numWindows > 1) {
            spectrogram.push_back(magnitudes);
            windowTimes.push_back(static_cast<double>(startIdx) / fmt.sampleRate);
        }
    }
    
    // Average the spectrum
    for (auto& mag : avgMagnitudes) {
        mag /= numWindows;
    }
    
    // Write averaged spectrum CSV
    std::filesystem::path specPath = outDir / (inName + "_spectrum_ch" + std::to_string(ch+1) + ".csv");
    auto frequencies = fft::getFrequencyBins(avgMagnitudes.size() * 2, fmt.sampleRate);
    
    wav::CsvWriter specCsv(specPath.string());
    specCsv.writeHeader("Frequency(Hz),Magnitude");
    for (size_t i = 0; i < frequencies.size() && i < avgMagnitudes.size(); ++i) {
        specCsv.writeRow(frequencies[i], avgMagnitudes[i]);
    }
    specCsv.close();
    
    std::string statusMsg = "Channel " + std::to_string(ch+1) + " done";
    
    // Write spectrogram if we have overlapping windows
    if (!spectrogram.empty()) {
        std::filesystem::path spectrogramPath = outDir / (inName + "_spectrogram_ch" + std::to_string(ch+1) + ".csv");
        
        wav::CsvWriter spectrogramCsv(spectrogramPath.string());
        spectrogramCsv.writeHeader("Time(s),Frequency(Hz),Magnitude");
        
        for (size_t w = 0; w < spectrogram.size(); ++w) {
            for (size_t f = 0; f < spectrogram[w].size() && f < frequencies.size(); ++f) {
                spectrogramCsv.writeRow(windowTimes[w], frequencies[f], spectrogram[w][f]);
            }
        }
        spectrogramCsv.close();
        
        statusMsg += " (with spectrogram)";
    }
    
    printThreadSafe(statusMsg);
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " input.wav output_dir [options]\n";
            std::cerr << "Options:\n";
            std::cerr << "  --window <type>       rectangular, hann, hamming, blackman (default: rectangular)\n";
            std::cerr << "  --overlap <percent>   Overlap percentage: 0-99 (default: 0)\n";
            std::cerr << "  --fft-size <samples>  FFT window size in samples (default: entire file)\n";
            std::cerr << "  --hop-size <samples>  Hop size in samples (overrides overlap)\n";
            std::cerr << "\nExamples:\n";
            std::cerr << "  " << argv[0] << " audio.wav output_dir --window hann --overlap 50\n";
            std::cerr << "  " << argv[0] << " audio.wav output_dir --window hamming --fft-size 2048 --overlap 75\n";
            return ERR_INVALID_INPUT;
        }

        std::filesystem::path inPath = argv[1];
        std::filesystem::path outDir = argv[2];
        std::string inName = inPath.stem().string();
        
        ProcessingOptions opts = parseCommandLine(argc, argv);

        // Create output directory
        if (!std::filesystem::exists(outDir)) {
            if (!std::filesystem::create_directories(outDir)) {
                std::cerr << "Failed to create output directory: " << outDir << '\n';
                return ERR_INVALID_INPUT;
            }
        }

        // Open and parse WAV file
        std::ifstream f(inPath, std::ios::binary);
        if (!f) { 
            std::cerr << "Failed to open input file: " << inPath << '\n'; 
            return ERR_FILE_NOT_FOUND; 
        }

        wav::WavFormat fmt;
        int result = wav::parseWavHeader(f, fmt);
        if (result != SUCCESS) return result;

        if (fmt.channels > MAX_NUM_CHANNELS) {
            std::cout << "Warning: File has " << fmt.channels 
                      << " channels. Processing only first " << MAX_NUM_CHANNELS << " channels.\n";
        }

        // Read audio data
        f.clear();
        f.seekg(fmt.dataPos);
        std::vector<uint8_t> raw(fmt.dataSize);
        if (!f.read(reinterpret_cast<char*>(raw.data()), fmt.dataSize)) {
            std::cerr << "Failed to read audio data\n";
            return ERR_READ_FAILURE;
        }

        size_t totalFrames = fmt.frames();
        std::cout << "Processing: " << fmt.channels << " channels, " 
                  << fmt.sampleRate << " Hz, " << fmt.bitsPerSample << "-bit, " 
                  << totalFrames << " frames\n";
        
        // Determine FFT window size for display
        size_t windowSize = (opts.fftSize > 0) ? opts.fftSize : totalFrames;
        if (windowSize > totalFrames) {
            windowSize = totalFrames;
        }
        size_t hopSize = opts.getHopSize(windowSize);
        
        std::cout << "FFT Configuration:\n";
        std::cout << "  Window size: " << windowSize << " samples\n";
        std::cout << "  Hop size: " << hopSize << " samples\n";
        std::cout << "  Overlap: " << (100.0 * (1.0 - (double)hopSize / windowSize)) << "%\n";

        // Launch async tasks per channel
        const size_t numChannelsToProcess = std::min(static_cast<size_t>(fmt.channels), 
                                                       static_cast<size_t>(MAX_NUM_CHANNELS));
        std::vector<std::future<void>> futures;
        
        for (size_t ch = 0; ch < numChannelsToProcess; ++ch) {
            futures.push_back(std::async(std::launch::async, processChannel, 
                                        ch, std::cref(raw), std::cref(fmt), std::cref(opts),
                                        std::cref(outDir), inName));
        }

        // Wait for all tasks and handle exceptions
        bool hadError = false;
        for (size_t i = 0; i < futures.size(); ++i) {
            try {
                futures[i].get();
            } catch (const std::exception& e) {
                printErrorThreadSafe("Error processing channel " + std::to_string(i+1) + ": " + e.what());
                hadError = true;
            } catch (int err) {
                printErrorThreadSafe("Error processing channel " + std::to_string(i+1) + ": code " + std::to_string(err));
                hadError = true;
            } catch (...) {
                printErrorThreadSafe("Unknown error processing channel " + std::to_string(i+1));
                hadError = true;
            }
        }

        if (hadError) {
            std::cerr << "Processing completed with errors\n";
            return ERR_FFT_COMPUTE;
        }

        std::cout << "All channels processed successfully!\n";
        return SUCCESS;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return ERR_READ_FAILURE;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred\n";
        return ERR_READ_FAILURE;
    }
}