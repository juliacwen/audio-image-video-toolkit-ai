/*
 * @file live_audio_denoise.cpp
 * @brief Live Audio Denoising with Multi-Profile Support (Desktop/Wearable/Embedded)
 * @author Julia Wen (wendigilane@gmail.com)
 * 
 * OVERVIEW:
 * Real-time audio denoising system using PortAudio and RNNoise with support for multiple build profiles. 
 * Features lock-free ring buffering and voice activity detection.
 * This is for local host only. For network streaming with send and receive, please look at live_audio_denoise_network.cpp.
 * 
 * Features:
 *  - Real-time audio input/output using PortAudio
 *  - Multi-channel support (1-16 channels depending on profile)
 *  - Lock-free Single-Producer Single-Consumer (SPSC) ring buffers for real-time safe audio streaming
 *  - Frame-based processing with RNNoise for denoising
 *  - Voice Activity Detection (VAD) for power saving
 *  - Profile-based optimization (Desktop/Wearable/Embedded)
 *  - Optional WAV recording for debugging purpose
 *  - RMS logging with periodic console output (CSV or text format)
 * 
 * BUILD PROFILES:
 * 
 * DESKTOP (default):
 *   - Purpose: Professional audio work, analysis, quality-first
 *   - Channels: Up to 16
 *   - Sample Rate: 48 kHz (RNNoise requirement)
 *   - Buffer: 1000ms (48000 frames)
 *   - VAD: Disabled by default (always processes)
 *   - Low Power: Disabled (full features)
 *   - WAV Recording: Enabled by default
 *   - Use case: Recording studios, content creation, research
 * 
 * WEARABLE:
 *   - Purpose: Battery-powered devices (phones, earbuds, AR/VR/XR)
 *   - Channels: Up to 8 (multi-mic arrays for spatial audio)
 *   - Sample Rate: 48 kHz (RNNoise requirement)
 *   - Buffer: 200ms (9600 frames) - lower latency
 *   - VAD: Enabled by default (power saving)
 *   - Low Power: Enabled by default
 *   - WAV Recording: Disabled by default (use --wav to enable)
 *   - Use case: Voice calls, AR/VR headsets, smart glasses
 * 
 * EMBEDDED:
 *   - Purpose: MCUs, IoT devices, minimal resources
 *   - Channels: 1 (mono only)
 *   - Sample Rate: 48 kHz (48kHz required by RNNoise, 16kHz would be better for embedded)
 *   - Buffer: 100ms (4800 frames) - minimal latency
 *   - VAD: Enabled by default (power saving)
 *   - Low Power: Enabled by default
 *   - WAV Recording: Disabled by default (use --wav to enable)
 *   - Use case: Smart speakers, intercom systems, IoT voice control
 * 
 * NOTE: RNNoise requires 48kHz sample rate. All profiles use 48kHz due to this. 
 * Using any other sample rate will produce poor quality or fail.
 * 
 * Dependencies:
 *  - PortAudio (https://www.portaudio.com/)
 *  - RNNoise library (https://github.com/xiph/rnnoise)
 *  - C++17 compiler with std::filesystem support
 * 
 * Build Instructions:
 *  CMake:
 *    cmake -DBUILD_PROFILE=DESKTOP ..
 *    make live_audio_denoise_desktop
 * 
 *    cmake -DBUILD_PROFILE=WEARABLE ..
 *    make live_audio_denoise_wearable
 * 
 *    cmake -DBUILD_PROFILE=EMBEDDED ..
 *    make live_audio_denoise_embedded
 * 
 * Configuration:
 *  Profile-specific constants are defined in inc/denoise_config.h
 * 
 * @par Revision History
 * - 11-19-2025 — Initial check-in (Julia Wen)
 * - 11-24-2025 — Lock-free Single-Producer Single-Consumer (SPSC) buffers
 * - 11-25-2025 — Add denormal control
 * - 12-01-2025 — Add bypass option and improvements
 * - 12-07-2025 — Add multi-profile support (Desktop/Wearable/Embedded)
 *                Add Voice Activity Detection (VAD)
 *                Add power optimization features
 *                Optimized RMS calculation
 *                Add --wav flag for debugging on wearable/embedded
 *                Add --no-low-power and --no-vad flags on wearable/embedded
 * - 01-14-2026 — Add CSV logging support for RMS data (--csv flag) 
 *                Mode 1: Without --csv Runtime: Writes rms_log.txt in space-separated format. On exit: Auto-converts to rms_log.csv
 *                Mode 2: With --csv Runtime: Writes rms_log.csv directly in CSV format
 * - 01-21-2026 — Refactored to use shared audio_framework_base
 */

#include <iostream>
#include <filesystem>
#include <thread>
#include <vector>
#include <cmath>
#include "../inc/audio_framework_base.h"
#include "../inc/denormal_control.h"
#include "../inc/denoise_config.h"
#include "rnnoise.h"

using namespace AudioFramework;

// ------------------ Denoise-Specific Context ------------------
struct DenoiseContext : public BaseAudioContext {
    std::vector<DenoiseState*> states;
    
    bool bypassDenoise;
    bool enableVAD;
    bool lowPowerMode;
    
    int vadHangoverCounter;
    bool isVoiceActive;
    float smoothedRMS;

    DenoiseContext(size_t bufferSize, int numCh, bool bypass, bool vad, bool lowPower, bool wavWrite, bool csvLog)
        : BaseAudioContext(bufferSize, numCh, wavWrite, csvLog),
          bypassDenoise(bypass),
          enableVAD(vad),
          lowPowerMode(lowPower),
          vadHangoverCounter(0),
          isVoiceActive(false),
          smoothedRMS(0.0f)
    {
        states.resize(numCh);
        for (int i = 0; i < numCh; ++i) {
            states[i] = rnnoise_create(nullptr);
            if (!states[i]) {
                throw std::runtime_error("Failed to create RNNoise state for channel " + std::to_string(i));
            }
        }
    }

    ~DenoiseContext() {
        for (auto* state : states) {
            if (state) rnnoise_destroy(state);
        }
    }
};

// ------------------ Processing Thread ------------------
void processingThread(DenoiseContext* ctx, int numChannels)
{
    denormal_control::AutoDisable autoDisable;

    std::vector<float> inFrame(FRAME_SIZE * numChannels, 0.0f);
    std::vector<float> outFrame(FRAME_SIZE * numChannels, 0.0f);
    std::vector<float> inCh(FRAME_SIZE);
    std::vector<float> outCh(FRAME_SIZE);

    const size_t totalSamplesNeeded = static_cast<size_t>(FRAME_SIZE * numChannels);

    size_t framesProcessed = 0;
    auto lastConsole = std::chrono::steady_clock::now();

    while (keepRunning) {
        while (ctx->inputBuffer.available() < totalSamplesNeeded) {
            if (!keepRunning) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
        }

        size_t got = 0;
        for (size_t i = 0; i < totalSamplesNeeded; i++) {
            float s;
            if (ctx->inputBuffer.pop(s)) {
                inFrame[i] = denormal_control::guardDenormal(s, ctx->denormalGuard);
                got++;
            } else break;
        }
        
        if (got < totalSamplesNeeded) continue;
        
        ctx->denormalGuard = -ctx->denormalGuard;

        float inRMS = calculateRMS(inFrame.data(), FRAME_SIZE, numChannels);
        
        bool processFrame = true;
        if (ctx->enableVAD) {
            const float alpha = 0.3f;
            ctx->smoothedRMS = alpha * inRMS + (1.0f - alpha) * ctx->smoothedRMS;

            if (ctx->smoothedRMS > VAD_THRESHOLD) {
                ctx->isVoiceActive = true;
                ctx->vadHangoverCounter = VAD_HANGOVER_FRAMES;
            } else if (ctx->vadHangoverCounter > 0) {
                ctx->vadHangoverCounter--;
                ctx->isVoiceActive = true;
            } else {
                ctx->isVoiceActive = false;
            }
            
            processFrame = ctx->isVoiceActive;
        }

        if (processFrame && !ctx->bypassDenoise) {
            for (int ch = 0; ch < numChannels; ++ch) {
                for (int i = 0; i < FRAME_SIZE; ++i) {
                    inCh[i] = inFrame[i * numChannels + ch];
                }

                rnnoise_process_frame(ctx->states[ch], outCh.data(), inCh.data());

                for (int i = 0; i < FRAME_SIZE; ++i) {
                    float sample = outCh[i];
                    if (sample > -DENORMAL_THRESHOLD && sample < DENORMAL_THRESHOLD) {
                        sample = 0.0f;
                    }
                    outFrame[i * numChannels + ch] = sample;
                }
            }
        } else {
            std::copy(inFrame.begin(), inFrame.end(), outFrame.begin());
        }

        ctx->outputBuffer.pushBulk(outFrame.data(), totalSamplesNeeded);

#if ENABLE_WAV_WRITING
        if (ctx->enableWavWrite) {
            for (int i = 0; i < FRAME_SIZE; ++i) {
                ctx->wavInput->writeFrame(&inFrame[i * numChannels], numChannels);
                ctx->wavOutput->writeFrame(&outFrame[i * numChannels], numChannels);
            }
        }
#endif

        framesProcessed++;

#if ENABLE_FILE_LOGGING
        if (framesProcessed % LOG_EVERY_N_FRAMES == 0) {
            float outRMS = calculateRMS(outFrame.data(), FRAME_SIZE, numChannels);
            
            if (ctx->enableCsvLog) {
                ctx->logFile << framesProcessed << "," 
                            << std::fixed << std::setprecision(6) << inRMS << ","
                            << std::fixed << std::setprecision(6) << outRMS << ","
                            << (processFrame ? 1 : 0) << "\n";
            } else {
                ctx->logFile << framesProcessed << " " << inRMS << " " << outRMS 
                            << " " << (processFrame ? 1 : 0) << "\n";
            }
        }
#endif

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastConsole).count() >= CONSOLE_INTERVAL_SEC) {
            float outRMS = calculateRMS(outFrame.data(), FRAME_SIZE, numChannels);
            std::cout << "[Frame " << framesProcessed << "] in_rms=" << inRMS
                      << ", out_rms=" << outRMS;
            if (ctx->enableVAD) {
                std::cout << ", vad=" << (ctx->isVoiceActive ? "active" : "idle");
            }
            std::cout << std::endl;
            lastConsole = now;
        }
    }
}

// ------------------ Main ------------------
int main(int argc, char* argv[])
{
    try {
        denormal_control::AutoDisable autoDisable;
        setupSignalHandler();

        // Parse common arguments
        CommonArgs commonArgs;
        if (!parseCommonArgs(argc, argv, commonArgs)) {
            return 1;
        }

        // Denoise-specific arguments
        bool bypassDenoise = false;
        bool enableVAD = ENABLE_VAD_DEFAULT;
        bool lowPowerMode = LOW_POWER_DEFAULT;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [options]\n\n";
                std::cout << "Options:\n";
                std::cout << "  --bypass         Bypass denoising\n";
                std::cout << "  --vad            Enable VAD\n";
                std::cout << "  --no-vad         Disable VAD\n";
                std::cout << "  --low-power      Enable low power mode\n";
                std::cout << "  --no-low-power   Disable low power mode\n";
                std::cout << "\nCommon options:\n";
                std::cout << "  -c N             Channels\n";
                std::cout << "  --wav            Enable WAV recording\n";
                std::cout << "  --csv            Enable CSV logging\n";
                std::cout << "  -o DIR           Output directory\n";
                return 0;
            }
            else if (arg == "--bypass") bypassDenoise = true;
            else if (arg == "--vad") enableVAD = true;
            else if (arg == "--no-vad") enableVAD = false;
            else if (arg == "--low-power") lowPowerMode = true;
            else if (arg == "--no-low-power") lowPowerMode = false;
        }

        if (!initPortAudio()) return 1;

        std::cout << "=== Audio Denoise ===\n";
        std::cout << "Channels: " << commonArgs.numChannels << "\n";
        std::cout << "VAD: " << (enableVAD ? "on" : "off") << "\n";
        std::cout << "Low power: " << (lowPowerMode ? "on" : "off") << "\n";
        std::cout << "WAV: " << (commonArgs.enableWavWrite ? "on" : "off") << "\n";
        std::cout << "CSV: " << (commonArgs.enableCsvLog ? "on" : "off") << "\n\n";

        if (!std::filesystem::exists(commonArgs.outputDir)) {
            std::filesystem::create_directories(commonArgs.outputDir);
        }

        size_t bufferSize = CIRCULAR_BUFFER_FRAMES * static_cast<size_t>(commonArgs.numChannels);
        DenoiseContext ctx(bufferSize, commonArgs.numChannels, bypassDenoise, enableVAD, 
                          lowPowerMode, commonArgs.enableWavWrite, commonArgs.enableCsvLog);

        ctx.initWavWriters(commonArgs.outputDir, "input_raw.wav", "output_denoised.wav", 
                          SAMPLE_RATE, commonArgs.bitDepth);
        
        ctx.initLogFile(commonArgs.outputDir, "rms_log", 
                       "frame,in_rms,out_rms,processed",
                       "frame in_rms out_rms processed");

        std::cout << "Press Ctrl+C to stop...\n\n";

        PaStream* stream = openAudioStream(commonArgs.numChannels, commonArgs.numChannels, 
                                          SAMPLE_RATE, FRAME_SIZE, 
                                          portAudioCallback, &ctx);
        if (!stream) {
            cleanupPortAudio();
            return 1;
        }

        std::thread procThread(processingThread, &ctx, commonArgs.numChannels);

        while (keepRunning) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "\nShutting down...\n";
        Pa_StopStream(stream);
        procThread.join();
        Pa_CloseStream(stream);
        cleanupPortAudio();

#if ENABLE_FILE_LOGGING
        ctx.logFile.close();
        if (!commonArgs.enableCsvLog) {
            ctx.convertLogToCSV(commonArgs.outputDir, "rms_log");
        }
#endif

        std::cout << "Done.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        cleanupPortAudio();
        return 1;
    }
}
