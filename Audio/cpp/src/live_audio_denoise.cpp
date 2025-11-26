/*
 * Live Audio Denoising Example using RNNoise and PortAudio
 * Author: Julia Wen (wendigilane@gmail.com)
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
 * - 2025-11-19 — Initial check-in  
 * - 2025-11-24 — Lock-free Single-Producer Single-Consumer (SPSC) buffers
 * - 2025-11-25 — Add denormal control
 */

#include <iostream>
#include <filesystem>
#include <csignal>
#include <atomic>
#include <thread>
#include <vector>
#include <cmath>
#include <fstream>
#include "../inc/wav_writer.h"
#include "../inc/SPSCFloatBuffer.h"
#include "../inc/denormal_control.h"
#include "rnnoise.h"
#include "portaudio.h"

// ------------------ Constants ------------------
constexpr int FRAME_SIZE = 480;
constexpr int SAMPLE_RATE = 48000;
constexpr int NUM_CHANNELS_DEFAULT = 1;
constexpr int CIRCULAR_BUFFER_FRAMES = 48000;
constexpr int CONSOLE_INTERVAL_SEC = 10;

// Denormal guard value (toggled each buffer to prevent DC buildup)
static float denormalGuard = 1.0e-20f;

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

    AudioIOContext(size_t bufferSize, int numChannels)
        : inputBuffer(bufferSize),
          outputBuffer(bufferSize)
    {
        states.resize(numChannels);
        for (int i = 0; i < numChannels; ++i) {
            states[i] = rnnoise_create(nullptr);
        }
    }

    ~AudioIOContext() {
        for (auto* state : states) {
            if (state) rnnoise_destroy(state);
        }
    }
};

// ------------------ PortAudio Callback ------------------
static int audioCallback([[maybe_unused]] const void* inputBuffer, 
                         void* outputBuffer,
                         unsigned long framesPerBuffer,
                         [[maybe_unused]] const PaStreamCallbackTimeInfo* timeInfo,
                         [[maybe_unused]] PaStreamCallbackFlags statusFlags,
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
    denormal_control::AutoDisable autoDisable;  // stays the same

    std::vector<float> inFrame(FRAME_SIZE * numChannels, 0.0f);
    std::vector<float> outFrame(FRAME_SIZE * numChannels, 0.0f);

    size_t framesProcessed = 0;
    auto lastConsole = std::chrono::steady_clock::now();

    while (keepRunning) {
        while (audioIOCtx->inputBuffer.available() < static_cast<size_t>(FRAME_SIZE * numChannels)) {
            if (!keepRunning) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        size_t got = 0;
        for (size_t i = 0; i < static_cast<size_t>(FRAME_SIZE * numChannels); i++) {
            float s;
            if (audioIOCtx->inputBuffer.pop(s)) {
                // Fully qualified namespace
                inFrame[i] = denormal_control::guardDenormal(s, denormalGuard);
                got++;
            }
        }
        if (got < static_cast<size_t>(FRAME_SIZE * numChannels)) continue;
        
        // Toggle denormal guard sign for next buffer
        denormalGuard = -denormalGuard;

        // Process each channel with its own state
        for (int ch = 0; ch < numChannels; ++ch) {
            std::vector<float> inCh(FRAME_SIZE);
            for (int i = 0; i < FRAME_SIZE; ++i)
                inCh[i] = inFrame[i * numChannels + ch];

            std::vector<float> outCh(FRAME_SIZE);
            rnnoise_process_frame(audioIOCtx->states[ch], outCh.data(), inCh.data());

            // Apply denormal guard to output as well
            for (int i = 0; i < FRAME_SIZE; ++i) {
                float sample = outCh[i];
                // Clamp very small values to zero
                if (sample > -1.0e-30f && sample < 1.0e-30f) {
                    sample = 0.0f;
                }
                outFrame[i * numChannels + ch] = sample;
            }
        }

        // Push output to buffer
        for (int i = 0; i < FRAME_SIZE * numChannels; i++)
            audioIOCtx->outputBuffer.push(outFrame[i]);

        // Write to WAV files
        for (int i = 0; i < FRAME_SIZE; ++i) {
            audioIOCtx->wavInput->writeFrame(&inFrame[i * numChannels]);
            audioIOCtx->wavOutput->writeFrame(&outFrame[i * numChannels]);
        }

        // Calculate RMS
        float inRMS = 0.0f, outRMS = 0.0f;
        for (int i = 0; i < FRAME_SIZE * numChannels; i += numChannels) {
            for (int ch = 0; ch < numChannels; ++ch) {
                inRMS  += inFrame[i + ch] * inFrame[i + ch];
                outRMS += outFrame[i + ch] * outFrame[i + ch];
            }
        }
        inRMS  = std::sqrt(inRMS / (FRAME_SIZE * numChannels));
        outRMS = std::sqrt(outRMS / (FRAME_SIZE * numChannels));

        framesProcessed++;
        audioIOCtx->logFile << framesProcessed << " " << inRMS << " " << outRMS << "\n";

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastConsole).count() >= CONSOLE_INTERVAL_SEC) {
            std::cout << "[Frame " << framesProcessed << "] in_rms=" << inRMS
                      << ", out_rms=" << outRMS << std::endl;
            lastConsole = now;
        }
    }
}

// ------------------ Main ------------------
int main(int argc, char* argv[])
{
    // Enable denormal handling, flush-to-zero / DAZ.
    denormal_control::AutoDisable autoDisable; // Constructor: calls disableDenormals() → sets FTZ/DAZ in CPU.

    std::signal(SIGINT, intHandler);

    std::filesystem::path outputDir = "test_output";
    int bitDepth = 16;
    int numChannels = NUM_CHANNELS_DEFAULT;

    if (argc >= 2) outputDir = argv[1];
    if (argc >= 3) bitDepth = std::stoi(argv[2]);
    if (argc >= 4) numChannels = std::stoi(argv[3]);

    Pa_Initialize();
    PaDeviceIndex inDev = Pa_GetDefaultInputDevice();
    const PaDeviceInfo* inInfo = Pa_GetDeviceInfo(inDev);
    if (numChannels > inInfo->maxInputChannels) numChannels = inInfo->maxInputChannels;

    PaDeviceIndex outDev = Pa_GetDefaultOutputDevice();
    const PaDeviceInfo* outInfo = Pa_GetDeviceInfo(outDev);
    if (numChannels > outInfo->maxOutputChannels) numChannels = outInfo->maxOutputChannels;

    if (!std::filesystem::exists(outputDir))
        std::filesystem::create_directories(outputDir);

    size_t bufferSize = CIRCULAR_BUFFER_FRAMES * static_cast<size_t>(numChannels);
    AudioIOContext audioIOCtx(bufferSize, numChannels);

    auto inputPath  = outputDir / "input_raw.wav";
    auto outputPath = outputDir / "output_denoised.wav";
    auto logPath    = outputDir / "rms_log.txt";

    audioIOCtx.wavInput  = std::make_unique<WavWriter>(inputPath.string(), SAMPLE_RATE, numChannels, bitDepth);
    audioIOCtx.wavOutput = std::make_unique<WavWriter>(outputPath.string(), SAMPLE_RATE, numChannels, bitDepth);
    audioIOCtx.logFile.open(logPath, std::ios::out);

#if defined(HAS_SSE_DENORMAL_CONTROL)
    std::cout << "Live denoising started (x86/x64 with SSE denormal control)\n";
#elif defined(HAS_ARM_DENORMAL_CONTROL)
    std::cout << "Live denoising started (ARM64 with FPU denormal control)\n";
#else
    std::cout << "Live denoising started (software denormal guard only)\n";
#endif
    std::cout << "Ctrl+C to stop...\n";

    PaStream* stream = nullptr;
    PaError err = Pa_OpenDefaultStream(
        &stream,
        numChannels,
        numChannels,
        paFloat32,
        SAMPLE_RATE,
        FRAME_SIZE,
        audioCallback,
        &audioIOCtx
    );
    if (err != paNoError) return 1;

    err = Pa_StartStream(stream);
    if (err != paNoError) return 1;

    std::thread procThread(processingThread, &audioIOCtx, numChannels);
    procThread.join();

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    audioIOCtx.logFile.close();

    std::cout << "Stopped. Saved " << inputPath << ", " << outputPath << ", " << logPath << "\n";
    return 0;
}