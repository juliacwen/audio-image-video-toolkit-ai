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
#include "rnnoise.h"
#include "portaudio.h"

// ------------------ Constants ------------------
constexpr int FRAME_SIZE = 480;
constexpr int SAMPLE_RATE = 48000;
constexpr int NUM_CHANNELS_DEFAULT = 1;
constexpr int CIRCULAR_BUFFER_FRAMES = 48000;  // 1 second buffer
constexpr int CONSOLE_INTERVAL_SEC = 10;

// ------------------ Signal Handling ------------------
std::atomic<bool> keepRunning{true};
void intHandler(int) { keepRunning.store(false); }

// ------------------ Audio IO Context ------------------
struct AudioIOContext {
    DenoiseState* st = nullptr;

    SPSCFloatBuffer inputBuffer;
    SPSCFloatBuffer outputBuffer;

    std::unique_ptr<WavWriter> wavInput;
    std::unique_ptr<WavWriter> wavOutput;

    std::ofstream logFile;

    AudioIOContext(size_t bufferSize)
        : inputBuffer(bufferSize),
          outputBuffer(bufferSize)
    {}
};

// ------------------ PortAudio Callback (Optimized for Real-Time Audio Processing with C++17) ------------------
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

    // Early exit if no input
    if (!in) return paContinue;

    // Calculate total samples (frames * channels)
    const auto numChannels = audioIOCtx->wavInput->getNumChannels();
    const auto totalSamples = framesPerBuffer * static_cast<size_t>(numChannels);

    // Push input samples using bulk operation (1 atomic op instead of N)
    audioIOCtx->inputBuffer.pushBulk(in, totalSamples);
    
    // Pop output samples using bulk operation (1 atomic op instead of N)
    const auto popped = audioIOCtx->outputBuffer.popBulk(out, totalSamples);
    
    // Fill remaining with silence if buffer underrun occurred
    // [[unlikely]] hints to compiler this is the exceptional path
    if (popped < totalSamples) {
        std::fill(out + popped, out + totalSamples, 0.0f);
    }

    return paContinue;
}

// ------------------ Processing Thread ------------------
void processingThread(AudioIOContext* audioIOCtx, int numChannels)
{
    std::vector<float> inFrame(FRAME_SIZE * numChannels);
    std::vector<float> outFrame(FRAME_SIZE * numChannels);

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
            if (audioIOCtx->inputBuffer.pop(s)) inFrame[i] = s, got++;
        }
        if (got < static_cast<size_t>(FRAME_SIZE * numChannels)) continue;

        for (int ch = 0; ch < numChannels; ++ch) {
            std::vector<float> inCh(FRAME_SIZE);
            for (int i = 0; i < FRAME_SIZE; ++i)
                inCh[i] = inFrame[i * numChannels + ch];

            std::vector<float> outCh(FRAME_SIZE);
            rnnoise_process_frame(audioIOCtx->st, outCh.data(), inCh.data());

            for (int i = 0; i < FRAME_SIZE; ++i)
                outFrame[i * numChannels + ch] = outCh[i];
        }

        for (int i = 0; i < FRAME_SIZE * numChannels; i++)
            audioIOCtx->outputBuffer.push(outFrame[i]);

        for (int i = 0; i < FRAME_SIZE; ++i) {
            audioIOCtx->wavInput->writeFrame(&inFrame[i * numChannels]);
            audioIOCtx->wavOutput->writeFrame(&outFrame[i * numChannels]);
        }

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
    AudioIOContext audioIOCtx(bufferSize);

    audioIOCtx.st = rnnoise_create(nullptr);
    if (!audioIOCtx.st) return 1;

    auto inputPath  = outputDir / "input_raw.wav";
    auto outputPath = outputDir / "output_denoised.wav";
    auto logPath    = outputDir / "rms_log.txt";

    audioIOCtx.wavInput  = std::make_unique<WavWriter>(inputPath.string(), SAMPLE_RATE, numChannels, bitDepth);
    audioIOCtx.wavOutput = std::make_unique<WavWriter>(outputPath.string(), SAMPLE_RATE, numChannels, bitDepth);
    audioIOCtx.logFile.open(logPath, std::ios::out);

    std::cout << "Live denoising started. Ctrl+C to stop...\n";

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
    rnnoise_destroy(audioIOCtx.st);
    audioIOCtx.logFile.close();

    std::cout << "Stopped. Saved " << inputPath << ", " << outputPath << ", " << logPath << "\n";
    return 0;
}

