/**
 * @file audio_framework_base.cpp
 * @brief Implementation of shared audio processing framework
 * @author Julia Wen (wendigilane@gmail.com)
 * Provides common infrastructure for real-time audio processing applications:
 * - PortAudio integration with lock-free buffers
 * - WAV recording support
 * - CSV/text logging
 * - Signal handling
 * - Command-line argument parsing
 * 
 * Used by:
 * - live_audio_denoise.cpp
 * - live_audio_tonal_detection.cpp
 * - Other real-time audio processing tool
 * 
 * @par Revision History
 * - 01-20-2026 — Initial creation - extracted from prior live_audio_denoise.cpp
 */

#include "../inc/audio_framework_base.h"
#include "../inc/denormal_control.h"
#include <csignal>
#include <cstring>
#include <algorithm>

namespace AudioFramework {

// ------------------ Signal Handling ------------------
std::atomic<bool> keepRunning{true};

static void intHandler(int) { 
    keepRunning.store(false); 
}

void setupSignalHandler() {
    std::signal(SIGINT, intHandler);
}

// ------------------ BaseAudioContext Implementation ------------------
BaseAudioContext::BaseAudioContext(size_t bufferSize, int numCh, bool wavWrite, bool csvLog)
    : inputBuffer(bufferSize),
      outputBuffer(bufferSize),
      enableWavWrite(wavWrite),
      enableCsvLog(csvLog),
      denormalGuard(DENORMAL_GUARD_INITIAL),
      numChannels(numCh)
{
}

bool BaseAudioContext::initWavWriters(const std::filesystem::path& outputDir,
                                     const std::string& inputName,
                                     const std::string& outputName,
                                     int sampleRate,
                                     int bitDepth)
{
#if ENABLE_WAV_WRITING
    if (!enableWavWrite) return false;
    
    try {
        if (!inputName.empty()) {
            auto inputPath = outputDir / inputName;
            wavInput = std::make_unique<WavWriter>(inputPath.string(), sampleRate, numChannels, bitDepth);
            std::cout << "[WAV] Recording input to " << inputPath << "\n";
        }
        
        if (!outputName.empty()) {
            auto outputPath = outputDir / outputName;
            wavOutput = std::make_unique<WavWriter>(outputPath.string(), sampleRate, numChannels, bitDepth);
            std::cout << "[WAV] Recording output to " << outputPath << "\n";
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

bool BaseAudioContext::initLogFile(const std::filesystem::path& outputDir,
                                  const std::string& logName,
                                  const std::string& csvHeader,
                                  const std::string& textHeader)
{
#if ENABLE_FILE_LOGGING
    std::string extension = enableCsvLog ? ".csv" : ".txt";
    auto logPath = outputDir / (logName + extension);
    logFilename = logPath.string();
    
    logFile.open(logPath, std::ios::out);
    if (!logFile.is_open()) {
        std::cerr << "[Log] Failed to open " << logPath << "\n";
        return false;
    }
    
    // Write appropriate header
    if (enableCsvLog) {
        logFile << csvHeader << "\n";
    } else {
        logFile << textHeader << "\n";
    }
    
    return true;
#else
    return false;
#endif
}

void BaseAudioContext::convertLogToCSV(const std::filesystem::path& /*outputDir*/,
                                      const std::string& /*logBaseName*/)
{
#if ENABLE_FILE_LOGGING
    // If already CSV format, nothing to do
    if (enableCsvLog) {
        std::cout << "[Log] Already in CSV format: " << logFilename << "\n";
        return;
    }
    
    // Close current log
    if (logFile.is_open()) {
        logFile.close();
    }
    
    // This is a placeholder - each application should override with custom conversion
    std::cout << "[Log] Auto-conversion to CSV not implemented for this log format\n";
#endif
}

// ------------------ PortAudio Callback ------------------
int portAudioCallback(const void* inputBuffer, 
                     void* outputBuffer,
                     unsigned long framesPerBuffer,
                     const PaStreamCallbackTimeInfo*,
                     PaStreamCallbackFlags,
                     void* userData)
{
    auto* ctx = static_cast<BaseAudioContext*>(userData);
    const auto* in = static_cast<const float*>(inputBuffer);
    auto* out = static_cast<float*>(outputBuffer);

    const auto numChannels = ctx->numChannels;
    const auto totalSamples = framesPerBuffer * static_cast<size_t>(numChannels);

    // Push input to buffer
    if (!in) {
        std::vector<float> silence(totalSamples, 0.0f);
        ctx->inputBuffer.pushBulk(silence.data(), totalSamples);
    } else {
        ctx->inputBuffer.pushBulk(in, totalSamples);
    }
    
    // Pop output from buffer
    const auto popped = ctx->outputBuffer.popBulk(out, totalSamples);
    if (popped < totalSamples) {
        std::fill(out + popped, out + totalSamples, 0.0f);
    }

    return paContinue;
}

// ------------------ Utility Functions ------------------
void convertToMono(const std::vector<float>& multiChannel, 
                  std::vector<float>& mono,
                  int numChannels,
                  int frameSize)
{
    mono.resize(frameSize);
    for (int i = 0; i < frameSize; ++i) {
        float sum = 0.0f;
        for (int ch = 0; ch < numChannels; ++ch) {
            sum += multiChannel[i * numChannels + ch];
        }
        mono[i] = sum / numChannels;
    }
}

bool initPortAudio()
{
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "[PortAudio] Init failed: " << Pa_GetErrorText(err) << "\n";
        return false;
    }
    return true;
}

void cleanupPortAudio()
{
    Pa_Terminate();
}

PaStream* openAudioStream(int inputChannels, 
                         int outputChannels,
                         int sampleRate,
                         int frameSize,
                         PaStreamCallback* callback,
                         void* userData)
{
    PaStream* stream = nullptr;
    
    PaError err = Pa_OpenDefaultStream(&stream, 
                                       inputChannels, 
                                       outputChannels, 
                                       paFloat32,
                                       sampleRate, 
                                       frameSize, 
                                       callback, 
                                       userData);
    
    if (err != paNoError) {
        std::cerr << "[PortAudio] Failed to open stream: " << Pa_GetErrorText(err) << "\n";
        return nullptr;
    }
    
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "[PortAudio] Failed to start stream: " << Pa_GetErrorText(err) << "\n";
        Pa_CloseStream(stream);
        return nullptr;
    }
    
    return stream;
}

bool parseCommonArgs(int& argc, char**& argv, CommonArgs& args)
{
    std::vector<char*> remainingArgs;
    remainingArgs.push_back(argv[0]); // Keep program name
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            args.outputDir = argv[++i];
        }
        else if ((arg == "-c" || arg == "--channels") && i + 1 < argc) {
            args.numChannels = std::stoi(argv[++i]);
        }
        else if ((arg == "-b" || arg == "--bitdepth") && i + 1 < argc) {
            args.bitDepth = std::stoi(argv[++i]);
            if (args.bitDepth != 16 && args.bitDepth != 24 && args.bitDepth != 32) {
                std::cerr << "Error: Bit depth must be 16, 24, or 32\n";
                return false;
            }
        }
        else if (arg == "--wav") {
            args.enableWavWrite = true;
        }
        else if (arg == "--csv") {
            args.enableCsvLog = true;
        }
        else {
            // Not a common arg, pass it through
            remainingArgs.push_back(argv[i]);
        }
    }
    
    // Update argc and argv to contain only non-common args
    argc = static_cast<int>(remainingArgs.size());
    for (size_t i = 0; i < remainingArgs.size(); ++i) {
        argv[i] = remainingArgs[i];
    }
    
    return true;
}

} // namespace AudioFramework