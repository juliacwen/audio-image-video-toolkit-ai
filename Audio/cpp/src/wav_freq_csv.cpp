/**
 * @file wav_freq_csv.cpp
 * @brief WAV audio file processor with windowing and FFT spectrum analysis
 * @author Julia Wen (wendigilane@gmail.com)
 * 
 * This program reads a WAV audio file (PCM 16/24-bit or Float 32-bit), applies
 * a chosing window function, and outputs:
 *   1. A CSV file containing the time-domain samples (windowed)
 *   2. A CSV file containing the frequency spectrum (FFT magnitude)
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
 * Usage: wav_freq_csv input.wav output.csv [channel] [window]
 *   channel: ignored (for backward compatibility with older versions)
 *   window: rectangular (default), hann, hamming, blackman
 * 
 * Output files:
 *   - output.csv: Index,Sample (windowed time-domain data)
 *   - output_spectrum.csv: Frequency(Hz),Magnitude (FFT spectrum)
 * 
 * @par Revision History
 * - 2025-09-07 — Initial check-in  
 * - 2025-11-12 — Added `error_codes.h`, error handling, improvement  
 * - 2025-11-21 — Code refactor  
 */
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include "../inc/error_codes.h"
#include "../inc/wav_utils.h"
#include "../inc/fft_utils.h"

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " input.wav output.csv [channel] [window]\n";
            std::cerr << "  channel: ignored for backward compatibility\n";
            std::cerr << "  window: rectangular (default), hann, hamming, blackman\n";
            return ERR_INVALID_INPUT;
        }

        std::string inPath = argv[1];
        std::string outPath = argv[2];
        
        // Parse window - could be at position 3 or 4 depending on if channel arg is present
        wav::WindowType window = wav::WindowType::Rectangular;
        if (argc >= 4) {
            std::string arg3 = argv[3];
            if (arg3.find_first_not_of("0123456789") == std::string::npos) {
                // It's a number, skip it and check argv[4] for window
                if (argc >= 5) {
                    window = wav::parseWindow(argv[4]);
                }
            } else {
                // It's the window name
                window = wav::parseWindow(arg3);
            }
        }

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

        size_t frames = fmt.frames();
        std::cout << "Processing: " << fmt.channels << " channel(s), " 
                  << fmt.sampleRate << " Hz, " << fmt.bitsPerSample << "-bit, " 
                  << frames << " frames\n";

        // Generate window coefficients
        auto windowCoeffs = wav::generateWindowCoeffs(window, frames);

        // Decode and window samples
        std::vector<double> samples(frames);
        for (size_t i = 0; i < frames; ++i) {
            samples[i] = wav::decodeSampleMono(raw, i, fmt) * windowCoeffs[i];
        }

        // Write time-domain CSV
        wav::CsvWriter csv(outPath);
        csv.writeHeader("Index,Sample");
        for (size_t i = 0; i < frames; ++i) {
            csv.writeRow(i, samples[i]);
        }
        csv.close();

        // Compute FFT
        auto fftData = fft::prepareForFFT(samples);
        if (fft::compute(fftData, false) != SUCCESS) {
            std::cerr << "FFT computation failed\n";
            return ERR_FFT_COMPUTE;
        }

        // Write spectrum CSV
        std::string specPath = outPath;
        if (auto dot = specPath.find_last_of('.'); dot != std::string::npos) {
            specPath.insert(dot, "_spectrum");
        } else {
            specPath += "_spectrum.csv";
        }

        auto magnitudes = fft::getMagnitudeSpectrum(fftData);
        auto frequencies = fft::getFrequencyBins(fftData.size(), fmt.sampleRate);

        wav::CsvWriter specCsv(specPath);
        specCsv.writeHeader("Frequency(Hz),Magnitude");
        for (size_t i = 0; i < frequencies.size(); ++i) {
            specCsv.writeRow(frequencies[i], magnitudes[i]);
        }
        specCsv.close();

        std::cout << "Successfully wrote " << samples.size() << " samples to " << outPath
                  << " and spectrum to " << specPath << '\n';

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
