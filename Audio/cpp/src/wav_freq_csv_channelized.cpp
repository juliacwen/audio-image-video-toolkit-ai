/**
 * @file wav_freq_csv_channelized.cpp
 * @brief Multi-channel WAV audio processor with windowing and FFT spectrum analysis
 * @author Julia Wen (wendigilane@gmail.com)
 * This program reads a multi-channel WAV audio file (PCM 16/24-bit or Float 32-bit),
 * applies an optional window function, and outputs per-channel:
 *   1. A CSV file containing the time-domain samples (windowed)
 *   2. A CSV file containing the frequency spectrum (FFT magnitude)
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
 * Usage: wav_freq_csv_channelized input.wav output_dir [window]
 *   window: rectangular (default), hann, hamming, blackman
 * 
 * Output files (per channel):
 *   - output_dir/filename_chN.csv: Index,Sample (windowed time-domain data)
 *   - output_dir/filename_spectrum_chN.csv: Frequency(Hz),Magnitude (FFT spectrum)
 * 
 * @par Revision History
 * - 11-21-2025 — Initial check-in  
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <future>
#include <mutex>
#include <filesystem>
#include "../inc/error_codes.h"
#include "../inc/wav_utils.h"
#include "../inc/fft_utils.h"

constexpr int MAX_NUM_CHANNELS = 14;

static std::mutex g_io_mutex;

void printThreadSafe(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_io_mutex);
    std::cout << msg << std::endl;
}

void printErrorThreadSafe(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_io_mutex);
    std::cerr << msg << std::endl;
}

void processChannel(size_t ch, const std::vector<uint8_t>& raw, 
                   const wav::WavFormat& fmt, wav::WindowType window,
                   const std::filesystem::path& outDir, const std::string& inName) {
    
    size_t frames = fmt.frames();
    
    // Generate window coefficients
    auto windowCoeffs = wav::generateWindowCoeffs(window, frames);
    
    // Decode and window samples for this channel
    std::vector<double> samples(frames);
    for (size_t i = 0; i < frames; ++i) {
        samples[i] = wav::decodeSampleChannel(raw, i, ch, fmt) * windowCoeffs[i];
    }
    
    // Write time-domain CSV
    std::filesystem::path csvPath = outDir / (inName + "_ch" + std::to_string(ch+1) + ".csv");
    wav::CsvWriter csv(csvPath.string());
    csv.writeHeader("Index,Sample");
    for (size_t i = 0; i < frames; ++i) {
        csv.writeRow(i, samples[i]);
    }
    csv.close();
    
    // Compute FFT
    auto fftData = fft::prepareForFFT(samples);
    if (fft::compute(fftData, false) != SUCCESS) {
        throw ERR_FFT_COMPUTE;
    }
    
    // Write spectrum CSV
    std::filesystem::path specPath = outDir / (inName + "_spectrum_ch" + std::to_string(ch+1) + ".csv");
    auto magnitudes = fft::getMagnitudeSpectrum(fftData);
    auto frequencies = fft::getFrequencyBins(fftData.size(), fmt.sampleRate);
    
    wav::CsvWriter specCsv(specPath.string());
    specCsv.writeHeader("Frequency(Hz),Magnitude");
    for (size_t i = 0; i < frequencies.size(); ++i) {
        specCsv.writeRow(frequencies[i], magnitudes[i]);
    }
    specCsv.close();
    
    printThreadSafe("Channel " + std::to_string(ch+1) + " done: " +
                    csvPath.string() + ", " + specPath.string());
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " input.wav output_dir [window]\n";
            std::cerr << "Window options: rectangular (default), hann, hamming, blackman\n";
            return ERR_INVALID_INPUT;
        }

        std::filesystem::path inPath = argv[1];
        std::filesystem::path outDir = argv[2];
        std::string inName = inPath.stem().string();
        wav::WindowType window = (argc >= 4) ? wav::parseWindow(argv[3]) : wav::WindowType::Rectangular;

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

        size_t frames = fmt.frames();
        std::cout << "Processing: " << fmt.channels << " channels, " 
                  << fmt.sampleRate << " Hz, " << fmt.bitsPerSample << "-bit, " 
                  << frames << " frames\n";

        // Launch async tasks per channel
        const size_t numChannelsToProcess = std::min(static_cast<size_t>(fmt.channels), 
                                                       static_cast<size_t>(MAX_NUM_CHANNELS));
        std::vector<std::future<void>> futures;
        
        for (size_t ch = 0; ch < numChannelsToProcess; ++ch) {
            futures.push_back(std::async(std::launch::async, processChannel, 
                                        ch, std::cref(raw), std::cref(fmt), window,
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
