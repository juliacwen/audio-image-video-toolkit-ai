/**
 * @file wav_freq_csv.cpp
 * @brief WAV audio file processor with windowing and overlapping FFT spectrum analysis
 * @author Julia Wen (wendigilane@gmail.com)
 * 
 * This program reads a WAV audio file (PCM 16/24-bit or Float 32-bit), applies
 * a chosen window function, and outputs:
 *   1. A CSV file containing the time-domain samples (windowed)
 *   2. A CSV file containing the frequency spectrum (FFT magnitude)
 *   3. Optional: Overlapping FFT analysis for better time-frequency resolution
 * 
 * For stereo or multi-channel files, the channels are mixed down to mono by
 * averaging the first two channels.
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
 * Usage: wav_freq_csv input.wav output.csv [options]
 *   Options:
 *     --window <type>       Window function: rectangular, hann, hamming, blackman (default: rectangular)
 *     --overlap <percent>   Overlap percentage: 0-99 (default: 0, no overlap)
 *     --fft-size <samples>  FFT window size in samples (default: entire file)
 *     --hop-size <samples>  Hop size in samples (overrides overlap percentage)
 * 
 * Output files:
 *   - output.csv: Index,Sample (windowed time-domain data for first window)
 *   - output_spectrum.csv: Frequency(Hz),Magnitude (averaged FFT spectrum from all windows)
 *   - output_spectrogram.csv (if overlapping): Time(s),Frequency(Hz),Magnitude
 * 
 * @par Revision History
 * - 09-07-2025 — Initial check-in  
 * - 11-12-2025 — Added `error_codes.h`, error handling, improvement  
 * - 11-21-2025 — Code refactor
 * - 12-04-2025 — Added overlapping FFT support
 */

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>
#include <vector>
#include <algorithm>
#include "../inc/error_codes.h"
#include "../inc/wav_utils.h"
#include "../inc/fft_utils.h"

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
        else if (arg.find_first_not_of("0123456789") == std::string::npos) {
            // Legacy: ignore numeric channel argument
            continue;
        }
        else if (arg.find("--") != 0) {
            // Legacy: window name without flag
            opts.window = wav::parseWindow(arg);
        }
    }
    
    return opts;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " input.wav output.csv [options]\n";
            std::cerr << "Options:\n";
            std::cerr << "  --window <type>       rectangular, hann, hamming, blackman (default: rectangular)\n";
            std::cerr << "  --overlap <percent>   Overlap percentage: 0-99 (default: 0)\n";
            std::cerr << "  --fft-size <samples>  FFT window size in samples (default: entire file)\n";
            std::cerr << "  --hop-size <samples>  Hop size in samples (overrides overlap)\n";
            std::cerr << "\nExamples:\n";
            std::cerr << "  " << argv[0] << " audio.wav output.csv --window hann --overlap 50\n";
            std::cerr << "  " << argv[0] << " audio.wav output.csv --window hamming --fft-size 2048 --overlap 75\n";
            return ERR_INVALID_INPUT;
        }

        std::string inPath = argv[1];
        std::string outPath = argv[2];
        
        ProcessingOptions opts = parseCommandLine(argc, argv);

        // Create output directory if needed
        std::filesystem::path outFilePath(outPath);
        if (outFilePath.has_parent_path()) {
            auto parentDir = outFilePath.parent_path();
            if (!parentDir.empty() && !std::filesystem::exists(parentDir)) {
                std::filesystem::create_directories(parentDir);
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

        if (fmt.channels > 2) {
            std::cout << "Warning: File has " << fmt.channels 
                      << " channels. Mixing down to mono using first 2 channels.\n";
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
        std::cout << "Processing: " << fmt.channels << " channel(s), " 
                  << fmt.sampleRate << " Hz, " << fmt.bitsPerSample << "-bit, " 
                  << totalFrames << " frames\n";

        // Decode all samples first
        std::vector<double> allSamples(totalFrames);
        for (size_t i = 0; i < totalFrames; ++i) {
            allSamples[i] = wav::decodeSampleMono(raw, i, fmt);
        }

        // Determine FFT window size
        size_t windowSize = (opts.fftSize > 0) ? opts.fftSize : totalFrames;
        if (windowSize > totalFrames) {
            std::cerr << "Warning: FFT size larger than file. Using entire file.\n";
            windowSize = totalFrames;
        }

        size_t hopSize = opts.getHopSize(windowSize);
        
        std::cout << "FFT Configuration:\n";
        std::cout << "  Window size: " << windowSize << " samples\n";
        std::cout << "  Hop size: " << hopSize << " samples\n";
        std::cout << "  Overlap: " << (100.0 * (1.0 - (double)hopSize / windowSize)) << "%\n";

        // Generate window coefficients
        auto windowCoeffs = wav::generateWindowCoeffs(opts.window, windowSize);

        // Calculate number of windows
        size_t numWindows = (totalFrames - windowSize) / hopSize + 1;
        if (hopSize > 0 && totalFrames >= windowSize) {
            std::cout << "  Number of windows: " << numWindows << "\n";
        } else {
            numWindows = 1;
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
                wav::CsvWriter csv(outPath);
                csv.writeHeader("Index,Sample");
                for (size_t i = 0; i < windowSize; ++i) {
                    csv.writeRow(i, windowedSamples[i]);
                }
                csv.close();
            }

            // Compute FFT
            auto fftData = fft::prepareForFFT(windowedSamples);
            if (fft::compute(fftData, false) != SUCCESS) {
                std::cerr << "FFT computation failed for window " << w << "\n";
                return ERR_FFT_COMPUTE;
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
                windowTimes.push_back((double)startIdx / fmt.sampleRate);
            }
        }

        // Average the spectrum
        for (auto& mag : avgMagnitudes) {
            mag /= numWindows;
        }

        // Write averaged spectrum CSV
        std::string specPath = outPath;
        if (auto dot = specPath.find_last_of('.'); dot != std::string::npos) {
            specPath.insert(dot, "_spectrum");
        } else {
            specPath += "_spectrum.csv";
        }

        auto frequencies = fft::getFrequencyBins(avgMagnitudes.size() * 2, fmt.sampleRate);

        wav::CsvWriter specCsv(specPath);
        specCsv.writeHeader("Frequency(Hz),Magnitude");
        for (size_t i = 0; i < frequencies.size() && i < avgMagnitudes.size(); ++i) {
            specCsv.writeRow(frequencies[i], avgMagnitudes[i]);
        }
        specCsv.close();

        std::cout << "Successfully wrote time-domain to " << outPath
                  << " and averaged spectrum to " << specPath << '\n';

        // Write spectrogram if we have overlapping windows
        if (!spectrogram.empty()) {
            std::string spectrogramPath = outPath;
            if (auto dot = spectrogramPath.find_last_of('.'); dot != std::string::npos) {
                spectrogramPath.insert(dot, "_spectrogram");
            } else {
                spectrogramPath += "_spectrogram.csv";
            }

            wav::CsvWriter spectrogramCsv(spectrogramPath);
            spectrogramCsv.writeHeader("Time(s),Frequency(Hz),Magnitude");
            
            for (size_t w = 0; w < spectrogram.size(); ++w) {
                for (size_t f = 0; f < spectrogram[w].size() && f < frequencies.size(); ++f) {
                    spectrogramCsv.writeRow(windowTimes[w], frequencies[f], spectrogram[w][f]);
                }
            }
            spectrogramCsv.close();
            
            std::cout << "Successfully wrote spectrogram to " << spectrogramPath << '\n';
        }

        return SUCCESS;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return ERR_READ_FAILURE;
    } catch (int err) {
        std::cerr << "Error code: " << err << '\n';
        return err;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred\n";
        return ERR_READ_FAILURE;
    }
}