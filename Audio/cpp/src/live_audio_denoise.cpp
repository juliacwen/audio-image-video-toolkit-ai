/*
 * @file live_audio_denoise.cpp
 * @brief Live Audio Denoising Example using RNNoise and PortAudio
 * @author Julia Wen (wendigilane@gmail.com)
 * Features:
 *  - Real-time audio input/output using PortAudio
 *  - Multi-channel support (up to MAX_CHANNELS)
 *  - Lock-free Single-Producer Single-Consumer (SPSC) buffers
 *    for real-time safe audio streaming
 *  - Frame-based processing with RNNoise for denoising
 *  - Saves input_raw.wav and output_denoised.wav
 *  - RMS logging to rms_log.txt (console every 10s)
 *
 * Thread Safety:
 *  - PortAudio callback is the producer, pushing audio into the input buffer
 *  - Processing thread is the consumer, reading from input buffer and writing
 *    to output buffer
 *  - Both buffers are lock-free SPSC queues to avoid blocking in the audio callback
 *  - `keepRunning` is an atomic flag for clean shutdown on Ctrl+C (SIGINT)
 *
 * Real-Time Considerations:
 *  - No mutexes or blocking operations are used in the audio callback
 *  - All disk I/O (WAV writing, logging) is done in the processing thread
 *  - Ensures low-latency, glitch-free audio streaming
 *
 * Denormal Handling:
 *  - Very small floating-point values (< ~1.175e-38 for float) are called denormals or subnormals
 *    which can cause severe CPU slowdown in DSP/audio code
 *  - DAZ (Denormals-Are-Zero) treats incoming denormal numbers as zero
 *  - FTZ (Flush-To-Zero) sets tiny results of calculations to zero
 *  - Enabling FTZ + DAZ prevents performance issues from denormals while preserving normal audio behavior
 *  - Implemented via `denormal_control::disableDenormals()` or RAII `denormal_control::AutoDisable`
 *  - Optional software guard `denormal_control::guardDenormal(value, guard)` clamps extremely small values

 * Usage:
 *  ./live_denoise [output_dir] [bit_depth] [num_channels]
 *
 * Dependencies:
 *  - PortAudio (https://www.portaudio.com/)
 *  - RNNoise library (https://github.com/xiph/rnnoise)
 *
 * @par Revision History
 * - 11-19-2025 — Initial check-in  
 * - 11-24-2025 — Lock-free Single-Producer Single-Consumer (SPSC) buffers
 * - 11-25-2025 — Add denormal control
 * - 12-01-2025 — Add bypass option and improvement
 * 
 */

#include <iostream>
#include <filesystem>
#include <csignal>
#include <atomic>
#include <thread>
#include <vector>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include "../inc/wav_writer.h"
#include "../inc/SPSCFloatBuffer.h"
#include "../inc/denormal_control.h"
#include "rnnoise.h"
#include "portaudio.h"

// ------------------ Constants ------------------
constexpr int FRAME_SIZE = 480;  // 10ms at 48kHz
constexpr int SAMPLE_RATE = 48000;
constexpr int NUM_CHANNELS_DEFAULT = 1;
constexpr int NUM_CHANNELS_MAX = 16;
constexpr int CIRCULAR_BUFFER_FRAMES = 48000;
constexpr int CONSOLE_INTERVAL_SEC = 10;
constexpr int POLL_INTERVAL_MS = 1;
constexpr float DENORMAL_THRESHOLD = 1.0e-30f;
constexpr float DENORMAL_GUARD_INITIAL = 1.0e-20f;

// ------------------ Signal Handling ------------------
std::atomic<bool> keepRunning{true};
void intHandler(int) { keepRunning.store(false); }

// ------------------ Audio IO Context ------------------
struct AudioIOContext {
    std::vector<DenoiseState*> states;

    SPSCFloatBuffer inputBuffer;
    SPSCFloatBuffer outputBuffer;

    std::unique_ptr<WavWriter> wavInput;
    std::unique_ptr<WavWriter> wavOutput;

    std::ofstream logFile;
    
    bool bypassDenoise;
    float denormalGuard;

    AudioIOContext(size_t bufferSize, int numChannels, bool bypass)
        : inputBuffer(bufferSize),
          outputBuffer(bufferSize),
          bypassDenoise(bypass),
          denormalGuard(DENORMAL_GUARD_INITIAL)
    {
        states.resize(numChannels);
        for (int i = 0; i < numChannels; ++i) {
            states[i] = rnnoise_create(nullptr);
            if (!states[i]) {
                throw std::runtime_error("Failed to create RNNoise state for channel " + std::to_string(i));
            }
        }
    }

    ~AudioIOContext() {
        for (auto* state : states) {
            if (state) rnnoise_destroy(state);
        }
    }
};

// ------------------ PortAudio Callback ------------------
static int audioCallback(const void* inputBuffer, 
                         void* outputBuffer,
                         unsigned long framesPerBuffer,
                         const PaStreamCallbackTimeInfo*,
                         PaStreamCallbackFlags,
                         void* userData) noexcept
{
    auto* audioIOCtx = static_cast<AudioIOContext*>(userData);
    const auto* in = static_cast<const float*>(inputBuffer);
    auto* out = static_cast<float*>(outputBuffer);

    const auto numChannels = audioIOCtx->wavInput->getNumChannels();
    const auto totalSamples = framesPerBuffer * static_cast<size_t>(numChannels);

    if (!in) {
        std::vector<float> silence(totalSamples, 0.0f);
        audioIOCtx->inputBuffer.pushBulk(silence.data(), totalSamples);
    } else {
        audioIOCtx->inputBuffer.pushBulk(in, totalSamples);
    }
    
    const auto popped = audioIOCtx->outputBuffer.popBulk(out, totalSamples);
    
    if (popped < totalSamples) {
        std::fill(out + popped, out + totalSamples, 0.0f);
    }

    return paContinue;
}

// ------------------ Processing Thread ------------------
void processingThread(AudioIOContext* audioIOCtx, int numChannels)
{
    // Enable hardware denormal handling
    denormal_control::AutoDisable autoDisable;

    // Pre-allocate buffers outside the loop to avoid repeated allocations
    std::vector<float> inFrame(FRAME_SIZE * numChannels, 0.0f);
    std::vector<float> outFrame(FRAME_SIZE * numChannels, 0.0f);
    std::vector<float> inCh(FRAME_SIZE);
    std::vector<float> outCh(FRAME_SIZE);

    const size_t totalSamplesNeeded = static_cast<size_t>(FRAME_SIZE * numChannels);

    size_t framesProcessed = 0;
    auto lastConsole = std::chrono::steady_clock::now();

    while (keepRunning) {
        // Wait for enough data with timeout to check keepRunning
        while (audioIOCtx->inputBuffer.available() < totalSamplesNeeded) {
            if (!keepRunning) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
        }

        // Read frame data
        size_t got = 0;
        for (size_t i = 0; i < totalSamplesNeeded; i++) {
            float s;
            if (audioIOCtx->inputBuffer.pop(s)) {
                inFrame[i] = denormal_control::guardDenormal(s, audioIOCtx->denormalGuard);
                got++;
            } else {
                break;
            }
        }
        
        if (got < totalSamplesNeeded) {
            // Incomplete frame - should rarely happen with proper buffering
            continue;
        }
        
        // Toggle denormal guard sign for next buffer
        audioIOCtx->denormalGuard = -audioIOCtx->denormalGuard;

        // Process each channel with its own state
        for (int ch = 0; ch < numChannels; ++ch) {
            // De-interleave channel
            for (int i = 0; i < FRAME_SIZE; ++i) {
                inCh[i] = inFrame[i * numChannels + ch];
            }

            // Always process through RNNoise
            rnnoise_process_frame(audioIOCtx->states[ch], outCh.data(), inCh.data());

            // Interleave output
            for (int i = 0; i < FRAME_SIZE; ++i) {
                float sample = outCh[i];
                
                // Clamp very small values to zero
                if (sample > -DENORMAL_THRESHOLD && sample < DENORMAL_THRESHOLD) {
                    sample = 0.0f;
                }
                
                // If bypassing, use original input instead of denoised output
                outFrame[i * numChannels + ch] = audioIOCtx->bypassDenoise ? inCh[i] : sample;
            }
        }

        // Push output to buffer using bulk operation
        audioIOCtx->outputBuffer.pushBulk(outFrame.data(), totalSamplesNeeded);

        // Write to WAV files (this is OK in non-real-time processing thread)
        for (int i = 0; i < FRAME_SIZE; ++i) {
            audioIOCtx->wavInput->writeFrame(&inFrame[i * numChannels], numChannels);
            audioIOCtx->wavOutput->writeFrame(&outFrame[i * numChannels], numChannels);
        }

        // Calculate RMS
        float inRMS = 0.0f, outRMS = 0.0f;
        for (size_t i = 0; i < totalSamplesNeeded; i += numChannels) {
            for (int ch = 0; ch < numChannels; ++ch) {
                inRMS  += inFrame[i + ch] * inFrame[i + ch];
                outRMS += outFrame[i + ch] * outFrame[i + ch];
            }
        }
        inRMS  = std::sqrt(inRMS / totalSamplesNeeded);
        outRMS = std::sqrt(outRMS / totalSamplesNeeded);

        framesProcessed++;
        audioIOCtx->logFile << framesProcessed << " " << inRMS << " " << outRMS << "\n";

        // Periodic console output
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastConsole).count() >= CONSOLE_INTERVAL_SEC) {
            std::cout << "[Frame " << framesProcessed << "] in_rms=" << inRMS
                      << ", out_rms=" << outRMS << std::endl;
            lastConsole = now;
        }
    }
}

// ------------------ Helper Functions ------------------
void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -o, --output DIR     Output directory (default: test_output)\n";
    std::cout << "  -b, --bitdepth N     WAV bit depth: 16, 24, or 32 (default: 16)\n";
    std::cout << "  -c, --channels N     Number of channels 1-" << NUM_CHANNELS_MAX << " (default: 1)\n";
    std::cout << "  --bypass             Bypass denoising (RNNoise still runs, output = input)\n";
    std::cout << "  -h, --help           Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << "\n";
    std::cout << "  " << programName << " --output my_recording --channels 2\n";
    std::cout << "  " << programName << " --bypass --bitdepth 24\n";
}

bool parseArguments(int argc, char* argv[], 
                   std::filesystem::path& outputDir,
                   int& bitDepth,
                   int& numChannels,
                   bool& bypassDenoise)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return false;
        }
        else if (arg == "--bypass") {
            bypassDenoise = true;
        }
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputDir = argv[++i];
        }
        else if ((arg == "-b" || arg == "--bitdepth") && i + 1 < argc) {
            try {
                bitDepth = std::stoi(argv[++i]);
                if (bitDepth != 16 && bitDepth != 24 && bitDepth != 32) {
                    std::cerr << "Error: Bit depth must be 16, 24, or 32\n";
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid bit depth value\n";
                return false;
            }
        }
        else if ((arg == "-c" || arg == "--channels") && i + 1 < argc) {
            try {
                numChannels = std::stoi(argv[++i]);
                if (numChannels < 1 || numChannels > NUM_CHANNELS_MAX) {
                    std::cerr << "Error: Number of channels must be between 1 and " 
                             << NUM_CHANNELS_MAX << "\n";
                    return false;
                }
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid channel count\n";
                return false;
            }
        }
        else {
            std::cerr << "Error: Unknown option '" << arg << "'\n";
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

// ------------------ Main ------------------
int main(int argc, char* argv[])
{
    try {
        // Enable denormal handling, flush-to-zero / DAZ
        denormal_control::AutoDisable autoDisable;
        
        std::signal(SIGINT, intHandler);

        // Default parameters
        std::filesystem::path outputDir = "test_output";
        int bitDepth = 16;
        int numChannels = NUM_CHANNELS_DEFAULT;
        bool bypassDenoise = false;

        // Parse command line arguments
        if (!parseArguments(argc, argv, outputDir, bitDepth, numChannels, bypassDenoise)) {
            return 0;  // Help was shown or error occurred
        }

        // Initialize PortAudio
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
            return 1;
        }

        // Get default devices
        PaDeviceIndex inDev = Pa_GetDefaultInputDevice();
        if (inDev == paNoDevice) {
            std::cerr << "Error: No default input device found\n";
            Pa_Terminate();
            return 1;
        }
        
        PaDeviceIndex outDev = Pa_GetDefaultOutputDevice();
        if (outDev == paNoDevice) {
            std::cerr << "Error: No default output device found\n";
            Pa_Terminate();
            return 1;
        }

        const PaDeviceInfo* inInfo = Pa_GetDeviceInfo(inDev);
        const PaDeviceInfo* outInfo = Pa_GetDeviceInfo(outDev);

        // Validate and adjust channel count
        int maxInputChannels = inInfo->maxInputChannels;
        int maxOutputChannels = outInfo->maxOutputChannels;
        int maxChannels = std::min(maxInputChannels, maxOutputChannels);
        
        if (numChannels > maxChannels) {
            std::cout << "Warning: Requested " << numChannels << " channels, but device supports max " 
                     << maxChannels << ". Using " << maxChannels << " channels.\n";
            numChannels = maxChannels;
        }

        std::cout << "Input device:  " << inInfo->name << " (max " << maxInputChannels << " ch)\n";
        std::cout << "Output device: " << outInfo->name << " (max " << maxOutputChannels << " ch)\n";
        std::cout << "Using: " << numChannels << " channel(s), " << SAMPLE_RATE 
                 << " Hz, " << bitDepth << "-bit\n";

        // Create output directory
        if (!std::filesystem::exists(outputDir)) {
            std::filesystem::create_directories(outputDir);
        }

        // Initialize audio context
        size_t bufferSize = CIRCULAR_BUFFER_FRAMES * static_cast<size_t>(numChannels);
        AudioIOContext audioIOCtx(bufferSize, numChannels, bypassDenoise);

        // Setup file paths
        auto inputPath  = outputDir / "input_raw.wav";
        auto outputPath = outputDir / "output_denoised.wav";
        auto logPath    = outputDir / "rms_log.txt";

        audioIOCtx.wavInput  = std::make_unique<WavWriter>(inputPath.string(), SAMPLE_RATE, numChannels, bitDepth);
        audioIOCtx.wavOutput = std::make_unique<WavWriter>(outputPath.string(), SAMPLE_RATE, numChannels, bitDepth);
        audioIOCtx.logFile.open(logPath, std::ios::out);
        
        if (!audioIOCtx.logFile.is_open()) {
            std::cerr << "Error: Could not open log file: " << logPath << std::endl;
            Pa_Terminate();
            return 1;
        }

        // Print status
#if defined(HAS_SSE_DENORMAL_CONTROL)
        std::cout << "Denormal control: x86/x64 SSE\n";
#elif defined(HAS_ARM_DENORMAL_CONTROL)
        std::cout << "Denormal control: ARM64 FPU\n";
#else
        std::cout << "Denormal control: Software guard only\n";
#endif
        
        if (bypassDenoise) {
            std::cout << "*** BYPASS MODE: Processing runs, but output = input ***\n";
        }
        
        std::cout << "Press Ctrl+C to stop...\n\n";

        // Open audio stream
        PaStream* stream = nullptr;
        err = Pa_OpenDefaultStream(
            &stream,
            numChannels,
            numChannels,
            paFloat32,
            SAMPLE_RATE,
            FRAME_SIZE,
            audioCallback,
            &audioIOCtx
        );
        
        if (err != paNoError) {
            std::cerr << "PortAudio error opening stream: " << Pa_GetErrorText(err) << std::endl;
            Pa_Terminate();
            return 1;
        }

        // Start audio stream
        err = Pa_StartStream(stream);
        if (err != paNoError) {
            std::cerr << "PortAudio error starting stream: " << Pa_GetErrorText(err) << std::endl;
            Pa_CloseStream(stream);
            Pa_Terminate();
            return 1;
        }

        // Start processing thread
        std::thread procThread(processingThread, &audioIOCtx, numChannels);

        // Wait for Ctrl+C
        while (keepRunning) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "\nShutting down...\n";

        // Stop audio stream
        err = Pa_StopStream(stream);
        if (err != paNoError) {
            std::cerr << "Warning: Error stopping stream: " << Pa_GetErrorText(err) << std::endl;
        }

        // Wait for processing thread to finish
        procThread.join();

        // Cleanup
        Pa_CloseStream(stream);
        Pa_Terminate();
        audioIOCtx.logFile.close();

        std::cout << "Stopped. Files saved:\n";
        std::cout << "  Input:  " << inputPath << "\n";
        std::cout << "  Output: " << outputPath << "\n";
        std::cout << "  Log:    " << logPath << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        Pa_Terminate();
        return 1;
    }
}