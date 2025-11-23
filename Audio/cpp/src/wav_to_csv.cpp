/**
 * @file wav_to_csv.cpp
 * @brief Simple WAV audio file to CSV converter
 * @author Julia Wen (wendigilane@gmail.com)
 * This program reads a WAV audio file (PCM 16/24-bit or Float 32-bit) and 
 * outputs the sample values to a CSV file. For stereo or multi-channel files,
 * the channels are mixed down to mono by averaging the first two channels.
 * 
 * Supported formats:
 *   - PCM 16-bit signed integer
 *   - PCM 24-bit signed integer
 *   - IEEE Float 32-bit
 * 
 * Usage: wav_to_csv input.wav output.csv [max_samples]
 *   max_samples: optional limit on number of samples to output (0 = all)
 * 
 * Output file format:
 *   - CSV with headers: Index,Sample
 *   - Each row: frame_index,sample_value
 * @par Revision History
 * - 2025-09-07 — Initial check-in  
 * - 2025-11-12 — Added `error_codes.h`, error handling, improvement  
 * - 2025-11-23 — Code refactor 
 */

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include "../inc/error_codes.h"
#include "../inc/wav_utils.h"

int main(int argc, char** argv) {
    try {
        if (argc < 3) { 
            std::cerr << "Usage: " << argv[0] << " input.wav output.csv [max_samples]\n";
            std::cerr << "  max_samples: optional limit (0 or omitted = all samples)\n";
            return ERR_INVALID_INPUT;
        }
        
        std::string inPath = argv[1];
        std::string outPath = argv[2];
        size_t max_samples = (argc >= 4) ? std::stoul(argv[3]) : 0;

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
            std::cerr << "Failed to open input file: " << inPath << "\n"; 
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
        size_t N = (max_samples == 0) ? frames : std::min(frames, max_samples);

        std::cout << "Processing: " << fmt.channels << " channel(s), " 
                  << fmt.sampleRate << " Hz, " << fmt.bitsPerSample << "-bit, " 
                  << N << " samples (of " << frames << " total)\n";

        // Write CSV
        wav::CsvWriter csv(outPath);
        csv.writeHeader("Index,Sample");

        for (size_t i = 0; i < N; ++i) {
            double sample = wav::decodeSampleMono(raw, i, fmt);
            csv.writeRow(i, sample);
        }

        csv.close();

        std::cout << "Successfully wrote " << N << " samples to " << outPath << "\n";
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
