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
 */

#include "../inc/audio_io_context.h"
#include "../inc/network_rtp.h"
#include "rnnoise.h"
#include <stdexcept>
#include <iostream>

#if ENABLE_WAV_WRITING
#include "../inc/wav_writer.h"
#endif

AudioIOContext::AudioIOContext(size_t bufferSize, int numCh, bool bypass, bool vad, bool lowPower, bool wavWrite)
    : inputBuffer(bufferSize), outputBuffer(bufferSize),
      bypassDenoise(bypass), enableVAD(vad), lowPowerMode(lowPower), enableWavWrite(wavWrite),
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
    logFile.open("test_output/rms_log.txt");
    if (logFile.is_open()) {
        logFile << "frame in_rms out_rms processed\n";
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

AudioIOContext::~AudioIOContext() {
    for (auto* state : states) {
        if (state) rnnoise_destroy(state);
    }
}