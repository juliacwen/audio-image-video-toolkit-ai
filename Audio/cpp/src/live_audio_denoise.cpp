/*live_audio_denoise.cpp
 * Live Audio Denoising with RNNoise and PortAudio (C++17)
 *
 * Features:
 *  - Thread-safe circular buffers for input/output
 *  - RNNoise processing per full frame
 *  - Saves input_raw.wav and output_denoised.wav
 *  - RMS logging to rms_log.txt (console every 10s)
 *  - Real-time safe PortAudio callback
 *
 * Requirements:
 *  - PortAudio (https://www.portaudio.com/)
 *  - RNNoise library (https://github.com/xiph/rnnoise)
 *
 * Author: Julia Wen (wendigilane@gmail.com)
 * Date: 11-19-2025
 */
#include <iostream>
#include <filesystem>
#include <csignal>
#include <atomic>
#include <thread>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "wav_writer.h"
#include "rnnoise.h" 
#include "portaudio.h"

// ------------------ Constants ------------------
constexpr int FRAME_SIZE = 480;
constexpr int SAMPLE_RATE = 48000;
constexpr int NUM_CHANNELS_DEFAULT = 1;
constexpr int MAX_CHANNELS = 14;
constexpr int CIRCULAR_BUFFER_FRAMES = 48000;  // 1 second buffer
constexpr int CONSOLE_INTERVAL_SEC = 10;

// ------------------ Signal Handling ------------------
std::atomic<bool> keepRunning{true};
void intHandler(int) { keepRunning.store(false); }

// ------------------ Thread-safe Circular Buffer ------------------
class CircularBuffer {
public:
    CircularBuffer(size_t size) : buffer(size, 0.0f), head(0), tail(0), count(0) {}

    // Copy and move are deleted (mutex prevents assignment)
    CircularBuffer(const CircularBuffer&) = delete;
    CircularBuffer& operator=(const CircularBuffer&) = delete;

    void push(float sample) {
        std::unique_lock<std::mutex> lock(mutex);
        buffer[head] = sample;
        head = (head + 1) % buffer.size();
        if (count < buffer.size()) count++;
        else tail = (tail + 1) % buffer.size(); // overwrite oldest if full
        cond.notify_one();
    }

    size_t pop(float* out, size_t n) {
        std::unique_lock<std::mutex> lock(mutex);
        size_t popped = 0;
        while (popped < n && count > 0) {
            out[popped++] = buffer[tail];
            tail = (tail + 1) % buffer.size();
            count--;
        }
        return popped;
    }

    size_t available() {
        std::unique_lock<std::mutex> lock(mutex);
        return count;
    }

    void wait_for(size_t n) {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [&]{ return count >= n || !keepRunning; });
    }

private:
    std::vector<float> buffer;
    size_t head, tail, count;
    std::mutex mutex;
    std::condition_variable cond;
};

// ------------------ Audio IO Context ------------------
struct AudioIOContext {
    DenoiseState* st = nullptr;

    CircularBuffer inputBuffer;
    CircularBuffer outputBuffer;

    std::unique_ptr<WavWriter> wavInput;
    std::unique_ptr<WavWriter> wavOutput;

    std::ofstream logFile;

    AudioIOContext(size_t bufferSize)
        : inputBuffer(bufferSize),
          outputBuffer(bufferSize)
    {}
};

// ------------------ PortAudio Callback ------------------
static int audioCallback(const void* inputBuffer, void* outputBuffer,
                         unsigned long framesPerBuffer,
                         const PaStreamCallbackTimeInfo*,
                         PaStreamCallbackFlags,
                         void* userData)
{
    auto* audioIOCtx = static_cast<AudioIOContext*>(userData);
    const float* in = static_cast<const float*>(inputBuffer);
    float* out = static_cast<float*>(outputBuffer);

    if (!in) return paContinue;

    int numChannels = audioIOCtx->wavInput->getNumChannels();
    for (unsigned long i = 0; i < framesPerBuffer * static_cast<unsigned long>(numChannels); i++)
        audioIOCtx->inputBuffer.push(in[i]);

    float temp;
    for (unsigned long i = 0; i < framesPerBuffer * static_cast<unsigned long>(numChannels); i++) {
        if (audioIOCtx->outputBuffer.pop(&temp, 1) == 1)
            out[i] = temp;
        else
            out[i] = 0.0f;
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
        audioIOCtx->inputBuffer.wait_for(FRAME_SIZE * numChannels);
        size_t got = audioIOCtx->inputBuffer.pop(inFrame.data(), FRAME_SIZE * numChannels);
        if (got < FRAME_SIZE * numChannels) continue;

        // Process each channel
        for (int ch = 0; ch < numChannels; ++ch) {
            std::vector<float> inCh(FRAME_SIZE);
            for (int i = 0; i < FRAME_SIZE; ++i)
                inCh[i] = inFrame[i * numChannels + ch];

            std::vector<float> outCh(FRAME_SIZE);

            rnnoise_process_frame(audioIOCtx->st, outCh.data(), inCh.data());

            for (int i = 0; i < FRAME_SIZE; ++i)
                outFrame[i * numChannels + ch] = outCh[i];
        }

        // Push to output buffer
        for (int i = 0; i < FRAME_SIZE * numChannels; ++i)
            audioIOCtx->outputBuffer.push(outFrame[i]);

        // Write interleaved frames to WAV
        for (int i = 0; i < FRAME_SIZE; ++i) {
            audioIOCtx->wavInput->writeFrame(&inFrame[i * numChannels]);
            audioIOCtx->wavOutput->writeFrame(&outFrame[i * numChannels]);
        }

        // RMS logging
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

    // Clamp channels to device max
    Pa_Initialize();
    PaDeviceIndex inDev = Pa_GetDefaultInputDevice();
    const PaDeviceInfo* inInfo = Pa_GetDeviceInfo(inDev);
    if (numChannels > inInfo->maxInputChannels) {
        std::cerr << "Warning: requested input channels " << numChannels
                  << " exceeds device max " << inInfo->maxInputChannels
                  << ". Reducing to max supported.\n";
        numChannels = inInfo->maxInputChannels;
    }

    PaDeviceIndex outDev = Pa_GetDefaultOutputDevice();
    const PaDeviceInfo* outInfo = Pa_GetDeviceInfo(outDev);
    if (numChannels > outInfo->maxOutputChannels) {
        std::cerr << "Warning: requested output channels " << numChannels
                  << " exceeds device max " << outInfo->maxOutputChannels
                  << ". Reducing to max supported.\n";
        numChannels = outInfo->maxOutputChannels;
    }
    Pa_Terminate();

    if (!std::filesystem::exists(outputDir)) {
        if (!std::filesystem::create_directories(outputDir)) {
            std::cerr << "ERROR: Could not create output directory: " << outputDir << "\n";
            return 1;
        }
    }

    size_t bufferSize = static_cast<size_t>(CIRCULAR_BUFFER_FRAMES) * static_cast<size_t>(numChannels);
    AudioIOContext audioIOCtx(bufferSize);

    audioIOCtx.st = rnnoise_create(nullptr);
    if (!audioIOCtx.st) {
        std::cerr << "ERROR: rnnoise_create failed\n";
        return 1;
    }

    auto inputPath  = outputDir / "input_raw.wav";
    auto outputPath = outputDir / "output_denoised.wav";
    auto logPath    = outputDir / "rms_log.txt";

    audioIOCtx.wavInput  = std::make_unique<WavWriter>(inputPath.string(), SAMPLE_RATE, numChannels, bitDepth);
    audioIOCtx.wavOutput = std::make_unique<WavWriter>(outputPath.string(), SAMPLE_RATE, numChannels, bitDepth);

    audioIOCtx.logFile.open(logPath, std::ios::out);

    std::cout << "Live denoising started. Ctrl+C to stop...\n";
    Pa_Initialize();
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
    if (err != paNoError) {
        std::cerr << "Pa_OpenDefaultStream error: " << Pa_GetErrorText(err) << "\n";
        return 1;
    }
    Pa_StartStream(stream);

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
