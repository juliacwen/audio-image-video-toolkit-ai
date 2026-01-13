/**
 * @file audio_io_context.cpp
 * @brief Implementation of AudioIOContext for audio processing pipeline
 * @author Julia Wen (wendigilane@gmail.com)
 * 
 * Provides initialization and cleanup for the audio context including:
 * - RNNoise state creation per channel
 * - WAV file writer initialization with mode-specific naming
 * - Optional file logging setup
 * 
 * @par Revision History
 * - 01-02-2026 — Initial Checkin
 * - 01-13-2026 - CSV conversion for RMS logs
 */

#include "../inc/audio_io_context.h"
#include "../inc/network_rtp.h"
#include "rnnoise.h"
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <iomanip>

#if ENABLE_WAV_WRITING
#include "../inc/wav_writer.h"
#endif

AudioIOContext::AudioIOContext(size_t bufferSize, int numCh, bool bypass, bool vad, bool lowPower, bool wavWrite, bool csvLog)
    : inputBuffer(bufferSize), outputBuffer(bufferSize),
      bypassDenoise(bypass), enableVAD(vad), lowPowerMode(lowPower), enableWavWrite(wavWrite), enableCsvLog(csvLog),
      denormalGuard(DENORMAL_GUARD_INITIAL), numChannels(numCh),
      vadHangoverCounter(0), isVoiceActive(false), smoothedRMS(0.0f)
{
    states.resize(numCh);
    for (int i = 0; i < numCh; ++i) {
        states[i] = rnnoise_create(nullptr);
        if (!states[i]) {
            throw std::runtime_error("Failed to create RNNoise state for channel " + std::to_string(i));
        }
    }

#if ENABLE_FILE_LOGGING
    logFilename = enableCsvLog ? "test_output/rms_log.csv" : "test_output/rms_log.txt";
    logFile.open(logFilename);
    if (logFile.is_open()) {
        if (enableCsvLog) {
            // CSV header
            logFile << "frame,in_rms,out_rms,processed\n";
        } else {
            // Space-separated header
            logFile << "frame in_rms out_rms processed\n";
        }
    }
#endif
}

bool AudioIOContext::initWavWriters(bool hasInput, bool hasOutput, bool isSender, bool isReceiver) {
#if ENABLE_WAV_WRITING
    if (!enableWavWrite) return false;
    
    try {
        if (hasInput) {
            wavInput = std::make_unique<WavWriter>("test_output/input_raw.wav", SAMPLE_RATE, numChannels);
            std::cout << "[WAV] Recording input to test_output/input_raw.wav\n";
        }
        if (hasOutput) {
            // Use different output filenames based on mode
            std::string outputFilename;
            if (isReceiver && !isSender) {
                outputFilename = "test_output/output_denoised_received.wav";
            } else if (isSender && !isReceiver) {
                outputFilename = "test_output/output_denoised_sent.wav";
            } else {
                outputFilename = "test_output/output_denoised.wav";
            }
            
            wavOutput = std::make_unique<WavWriter>(outputFilename, SAMPLE_RATE, numChannels);
            std::cout << "[WAV] Recording output to " << outputFilename << "\n";
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[WAV] Failed to initialize: " << e.what() << "\n";
        enableWavWrite = false;
        return false;
    }
#else
    if (enableWavWrite) {
        std::cout << "[WAV] WAV writing is disabled in this build\n";
        enableWavWrite = false;
    }
    return false;
#endif
}

void AudioIOContext::convertLogToCSV() {
#if ENABLE_FILE_LOGGING
    // If already CSV format, nothing to do
    if (enableCsvLog) {
        std::cout << "[RMS Log] Already in CSV format: " << logFilename << "\n";
        return;
    }
    
    // Close the current log file
    if (logFile.is_open()) {
        logFile.close();
    }
    
    // Read the space-separated file
    std::ifstream inFile("test_output/rms_log.txt");
    if (!inFile.is_open()) {
        std::cerr << "[RMS Log] Could not open rms_log.txt for conversion\n";
        return;
    }
    
    // Open CSV output file
    std::ofstream csvFile("test_output/rms_log.csv");
    if (!csvFile.is_open()) {
        std::cerr << "[RMS Log] Could not create rms_log.csv\n";
        inFile.close();
        return;
    }
    
    // Write CSV header
    csvFile << "frame,in_rms,out_rms,processed\n";
    
    std::string line;
    bool firstLine = true;
    int lineCount = 0;
    
    while (std::getline(inFile, line)) {
        // Skip the header line
        if (firstLine) {
            firstLine = false;
            continue;
        }
        
        // Parse space-separated values
        std::istringstream iss(line);
        uint64_t frame;
        float inRMS, outRMS;
        int processed;
        
        if (iss >> frame >> inRMS >> outRMS >> processed) {
            // Write as CSV
            csvFile << frame << "," 
                    << std::fixed << std::setprecision(6) << inRMS << ","
                    << std::fixed << std::setprecision(6) << outRMS << ","
                    << processed << "\n";
            lineCount++;
        }
    }
    
    inFile.close();
    csvFile.close();
    
    std::cout << "[RMS Log] Converted " << lineCount << " lines to CSV: test_output/rms_log.csv\n";
#else
    std::cout << "[RMS Log] File logging is disabled in this build\n";
#endif
}

AudioIOContext::~AudioIOContext() {
    for (auto* state : states) {
        if (state) rnnoise_destroy(state);
    }
    
#if ENABLE_FILE_LOGGING
    if (logFile.is_open()) {
        logFile.close();
    }
    
    // Auto-convert to CSV if not already in CSV format
    if (!enableCsvLog) {
        convertLogToCSV();
    }
#endif
}