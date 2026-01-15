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
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <csignal>
#include <atomic>
#include <thread>
#include <vector>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include "../inc/denoise_config.h"
#include "../inc/SPSCFloatBuffer.h"
#include "../inc/denormal_control.h"
#include "rnnoise.h"
#include "portaudio.h"

#if ENABLE_WAV_WRITING
#include "../inc/wav_writer.h"
#endif

// ------------------ Signal Handling ------------------
std::atomic<bool> keepRunning{true};
void intHandler(int) { keepRunning.store(false); }

// ------------------ Audio IO Context ------------------
/**
 * AudioIOContext - Central state for audio processing
 * 
 * Contains all state needed for real-time audio processing including
 * RNNoise states, buffers, file writers, and VAD state.
 * 
 * Thread Safety:
 * - states: Read-only after initialization (thread-safe)
 * - inputBuffer/outputBuffer: Lock-free SPSC ring buffer(thread-safe by design)
 * - VAD state: Only accessed by processing thread (no sync needed)
 * - wavInput/wavOutput: Only accessed by processing thread
 */
struct AudioIOContext {
    // RNNoise state (one per channel)
    std::vector<DenoiseState*> states;
    
    // Lock-free circular buffers for audio data
    SPSCFloatBuffer inputBuffer;   // PortAudio callback → Processing thread
    SPSCFloatBuffer outputBuffer;  // Processing thread → PortAudio callback

#if ENABLE_WAV_WRITING
    std::unique_ptr<WavWriter> wavInput;   // Records raw input audio
    std::unique_ptr<WavWriter> wavOutput;  // Records denoised output audio
#endif

#if ENABLE_FILE_LOGGING
    std::ofstream logFile;  // RMS and VAD state log
    std::string logFilename;  // Store filename for CSV conversion
#endif
    
    // Configuration flags
    bool bypassDenoise;   // If true, output = input (for testing)
    bool enableVAD;       // Voice Activity Detection enabled
    bool lowPowerMode;    // Reduced logging and I/O
    bool enableWavWrite;  // Enable WAV file writing (separate from low power)
    bool enableCsvLog;    // Enable CSV format for RMS logging
    float denormalGuard;  // Alternating offset to prevent denormals
    int numChannels;      // Number of audio channels
    
    // VAD state (Voice Activity Detection)
    int vadHangoverCounter;   // Frames remaining to keep processing after voice stops
    bool isVoiceActive;       // Current VAD decision
    float smoothedRMS;        // Exponentially smoothed RMS for stable VAD

    AudioIOContext(size_t bufferSize, int numCh, bool bypass, bool vad, bool lowPower, bool wavWrite, bool csvLog)
        : inputBuffer(bufferSize),
          outputBuffer(bufferSize),
          bypassDenoise(bypass),
          enableVAD(vad),
          lowPowerMode(lowPower),
          enableWavWrite(wavWrite),
          enableCsvLog(csvLog),
          denormalGuard(DENORMAL_GUARD_INITIAL),
          numChannels(numCh),
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

    ~AudioIOContext() {
        for (auto* state : states) {
            if (state) rnnoise_destroy(state);
        }
    }
    
    void convertLogToCSV(const std::filesystem::path& outputDir) {
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
        auto txtPath = outputDir / "rms_log.txt";
        std::ifstream inFile(txtPath);
        if (!inFile.is_open()) {
            std::cerr << "[RMS Log] Could not open rms_log.txt for conversion\n";
            return;
        }
        
        // Open CSV output file
        auto csvPath = outputDir / "rms_log.csv";
        std::ofstream csvFile(csvPath);
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
        
        std::cout << "[RMS Log] Converted " << lineCount << " lines to CSV: " << csvPath << "\n";
#endif
    }
};

// ------------------ PortAudio Callback ------------------
/**
 * audioCallback - Real-time audio I/O callback (runs on PortAudio thread)
 * 
 * This function runs in a real-time audio thread with strict timing requirements.
 * MUST NOT: Allocate memory, acquire locks, do I/O, or call blocking functions.
 * 
 * Input path: Copy audio from PortAudio → Input buffer (lock-free push)
 * Output path: Copy audio from Output buffer → PortAudio (lock-free pop)
 * 
 * If output buffer underruns, fills with silence to prevent glitches.
 * 
 * @param inputBuffer Raw audio from microphone (may be NULL)
 * @param outputBuffer Destination buffer for speaker
 * @param framesPerBuffer Number of frames to process
 * @param userData Pointer to AudioIOContext
 * @return paContinue to keep stream running
 */
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

    const auto numChannels = audioIOCtx->numChannels;
    const auto totalSamples = framesPerBuffer * static_cast<size_t>(numChannels);

    // Push input to buffer (silence if no input)
    if (!in) {
        std::vector<float> silence(totalSamples, 0.0f);
        audioIOCtx->inputBuffer.pushBulk(silence.data(), totalSamples);
    } else {
        audioIOCtx->inputBuffer.pushBulk(in, totalSamples);
    }
    
    // Pop output from buffer (silence on underrun)
    const auto popped = audioIOCtx->outputBuffer.popBulk(out, totalSamples);
    
    if (popped < totalSamples) {
        // Buffer underrun - fill remaining with silence to prevent glitches
        std::fill(out + popped, out + totalSamples, 0.0f);
    }

    return paContinue;
}

// ------------------ Fast RMS Calculation ------------------
/**
 * calculateRMS - Optimized Root Mean Square calculation
 * 
 * Computes RMS with loop unrolling for better performance.
 * Processes 4 samples at a time to leverage CPU pipelining.
 * 
 * @param data Audio sample array
 * @param count Number of samples to process
 * @param stride Step between samples (1 for mono, numChannels for interleaved)
 * @return RMS value (0.0 = silence, 1.0 = full scale)
 */
inline float calculateRMS(const float* data, size_t count, int stride = 1) {
    float sum = 0.0f;
    size_t i = 0;
    size_t count4 = (count / 4) * 4;
    
    // Unrolled loop: process 4 samples at once
    for (; i < count4; i += 4) {
        float s0 = data[i * stride];
        float s1 = data[(i + 1) * stride];
        float s2 = data[(i + 2) * stride];
        float s3 = data[(i + 3) * stride];
        sum += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
    }
    
    // Handle remaining samples
    for (; i < count; ++i) {
        float s = data[i * stride];
        sum += s * s;
    }
    
    return std::sqrt(sum / count);
}

// ------------------ Processing Thread ------------------
/**
 * processingThread - Main audio processing worker
 * 
 * This thread runs continuously, consuming frames from the input buffer,
 * processing them through RNNoise, and pushing results to the output buffer.
 * 
 * PROCESSING PIPELINE:
 * 1. Poll input buffer for complete frame
 * 2. Apply denormal guard (alternating small offset)
 * 3. Calculate RMS and check VAD (if enabled)
 * 4. For each channel:
 *    a. De-interleave (extract single channel)
 *    b. Process through RNNoise (if VAD active or VAD disabled)
 *    c. Clamp denormals to zero
 *    d. Interleave back to multi-channel
 * 5. Push to output buffer
 * 6. Write to WAV files (if enabled)
 * 7. Log RMS values (if enabled)
 * 
 * VOICE ACTIVITY DETECTION:
 * - Smoothed RMS compared to threshold
 * - Hangover counter prevents choppy audio on brief pauses
 * - When inactive, audio passes through unprocessed (saves power)
 * 
 * @param audioIOCtx Shared context with buffers and state
 * @param numChannels Number of audio channels to process
 */
void processingThread(AudioIOContext* audioIOCtx, int numChannels)
{
    // Enable hardware denormal handling (FTZ/DAZ)
    denormal_control::AutoDisable autoDisable;

    // Pre-allocate buffers (reused every frame)
    std::vector<float> inFrame(FRAME_SIZE * numChannels, 0.0f);
    std::vector<float> outFrame(FRAME_SIZE * numChannels, 0.0f);
    std::vector<float> inCh(FRAME_SIZE);   // Single channel working buffer
    std::vector<float> outCh(FRAME_SIZE);  // Single channel output buffer

    const size_t totalSamplesNeeded = static_cast<size_t>(FRAME_SIZE * numChannels);

    size_t framesProcessed = 0;
    auto lastConsole = std::chrono::steady_clock::now();

    while (keepRunning) {
        // Wait for enough data (non-blocking poll with timeout)
        while (audioIOCtx->inputBuffer.available() < totalSamplesNeeded) {
            if (!keepRunning) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
        }

        // Read complete frame from input buffer
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
        
        if (got < totalSamplesNeeded) continue;  // Incomplete frame, retry
        
        // Toggle denormal guard sign (prevents systematic bias)
        audioIOCtx->denormalGuard = -audioIOCtx->denormalGuard;

        // Calculate input RMS (first channel only for efficiency)
        float inRMS = calculateRMS(inFrame.data(), FRAME_SIZE, numChannels);
        
        // Voice Activity Detection
        bool processFrame = true;
        if (audioIOCtx->enableVAD) {
            // Exponential smoothing: smooth = α*new + (1-α)*old
            const float alpha = 0.3f;  // Smoothing factor (0=no change, 1=instant)
            audioIOCtx->smoothedRMS = alpha * inRMS + (1.0f - alpha) * audioIOCtx->smoothedRMS;

            // VAD decision with hangover
            if (audioIOCtx->smoothedRMS > VAD_THRESHOLD) {
                // Voice detected: activate and reset hangover
                audioIOCtx->isVoiceActive = true;
                audioIOCtx->vadHangoverCounter = VAD_HANGOVER_FRAMES;
            } else if (audioIOCtx->vadHangoverCounter > 0) {
                // In hangover period: keep processing
                audioIOCtx->vadHangoverCounter--;
                audioIOCtx->isVoiceActive = true;
            } else {
                // No voice, hangover expired: deactivate
                audioIOCtx->isVoiceActive = false;
            }
            
            processFrame = audioIOCtx->isVoiceActive;
        }

        // Process each channel through RNNoise
        if (processFrame && !audioIOCtx->bypassDenoise) {
            for (int ch = 0; ch < numChannels; ++ch) {
                // De-interleave: extract single channel
                for (int i = 0; i < FRAME_SIZE; ++i) {
                    inCh[i] = inFrame[i * numChannels + ch];
                }

                // Process through RNNoise (spectral subtraction + ML)
                rnnoise_process_frame(audioIOCtx->states[ch], outCh.data(), inCh.data());

                // Interleave: merge back into multi-channel frame
                for (int i = 0; i < FRAME_SIZE; ++i) {
                    float sample = outCh[i];
                    
                    // Clamp denormals to zero (final safety net)
                    if (sample > -DENORMAL_THRESHOLD && sample < DENORMAL_THRESHOLD) {
                        sample = 0.0f;
                    }
                    
                    outFrame[i * numChannels + ch] = sample;
                }
            }
        } else {
            // Bypass: pass through unprocessed (VAD inactive or bypass mode)
            std::copy(inFrame.begin(), inFrame.end(), outFrame.begin());
        }

        // Push to output buffer (lock-free)
        audioIOCtx->outputBuffer.pushBulk(outFrame.data(), totalSamplesNeeded);

#if ENABLE_WAV_WRITING
        // Write to WAV files (only if WAV writing is enabled)
        if (audioIOCtx->enableWavWrite) {
            for (int i = 0; i < FRAME_SIZE; ++i) {
                audioIOCtx->wavInput->writeFrame(&inFrame[i * numChannels], numChannels);
                audioIOCtx->wavOutput->writeFrame(&outFrame[i * numChannels], numChannels);
            }
        }
#endif

        framesProcessed++;

#if ENABLE_FILE_LOGGING
        // Log RMS values periodically (not every frame in wearable/embedded)
        if (framesProcessed % LOG_EVERY_N_FRAMES == 0) {
            float outRMS = calculateRMS(outFrame.data(), FRAME_SIZE, numChannels);
            
            if (audioIOCtx->enableCsvLog) {
                // CSV format: frame,in_rms,out_rms,processed
                audioIOCtx->logFile << framesProcessed << "," 
                                   << std::fixed << std::setprecision(6) << inRMS << ","
                                   << std::fixed << std::setprecision(6) << outRMS << ","
                                   << (processFrame ? 1 : 0) << "\n";
            } else {
                // Space-separated format
                audioIOCtx->logFile << framesProcessed << " " << inRMS << " " << outRMS 
                                   << " " << (processFrame ? 1 : 0) << "\n";
            }
        }
#endif

        // Console output (periodic, profile-dependent frequency)
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastConsole).count() >= CONSOLE_INTERVAL_SEC) {
            float outRMS = calculateRMS(outFrame.data(), FRAME_SIZE, numChannels);
            std::cout << "[Frame " << framesProcessed << "] in_rms=" << inRMS
                      << ", out_rms=" << outRMS;
            if (audioIOCtx->enableVAD) {
                std::cout << ", vad=" << (audioIOCtx->isVoiceActive ? "active" : "idle");
            }
            std::cout << std::endl;
            lastConsole = now;
        }
    }
}

// ------------------ Helper Functions ------------------
void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n\n";
    
#ifdef BUILD_WEARABLE
    std::cout << "Build profile: WEARABLE (low power, 8ch max, VAD enabled by default)\n";
    std::cout << "Optimized for: AR/VR/XR headsets, smart glasses, phones\n\n";
#elif defined(BUILD_EMBEDDED)
    std::cout << "Build profile: EMBEDDED (minimal resources, 1ch mono, VAD enabled by default)\n";
    std::cout << "Optimized for: IoT devices, smart speakers, intercoms\n\n";
#else
    std::cout << "Build profile: DESKTOP (full features, 16ch max, VAD disabled by default)\n";
    std::cout << "Optimized for: Studio recording, conference rooms\n\n";
#endif

    std::cout << "Options:\n";
    
#if ENABLE_WAV_WRITING
    std::cout << "  -o, --output DIR     Output directory (default: test_output)\n";
    std::cout << "  -b, --bitdepth N     WAV bit depth: 16, 24, or 32 (default: 16)\n";
#endif
    
    std::cout << "  -c, --channels N     Number of channels 1-" << NUM_CHANNELS_MAX << " (default: 1)\n";
    std::cout << "  --bypass             Bypass denoising (pass-through)\n";
    std::cout << "  --csv                Enable CSV format for RMS logging (default: text)\n";
    
    // Show appropriate VAD option based on default
    if (ENABLE_VAD_DEFAULT) {
        std::cout << "  --no-vad             Disable Voice Activity Detection (enabled by default)\n";
    } else {
        std::cout << "  --vad                Enable Voice Activity Detection (disabled by default)\n";
    }

    // Show appropriate low-power option based on default
    if (LOW_POWER_DEFAULT) {
        std::cout << "  --no-low-power       Disable low power mode (enabled by default)\n";
        std::cout << "  --wav                Enable WAV recording (disabled in low power mode)\n";
    } else {
        std::cout << "  --low-power          Enable low power mode (disabled by default)\n";
    }
    
    std::cout << "  -h, --help           Show this help\n\n";
    
    std::cout << "Examples:\n";
#ifdef BUILD_WEARABLE
    std::cout << "  " << programName << "                           # Default: VAD on, low power on, no WAV\n";
    std::cout << "  " << programName << " --channels 8               # 8-channel (AR/VR)\n";
    std::cout << "  " << programName << " --wav --csv                # Enable WAV and CSV logging\n";
    std::cout << "  " << programName << " --no-vad --no-low-power    # Full power, no optimizations\n";
#elif defined(BUILD_EMBEDDED)
    std::cout << "  " << programName << "                           # Default: mono, VAD on, low power on, no WAV\n";
    std::cout << "  " << programName << " --wav --csv                # Enable WAV and CSV logging\n";
    std::cout << "  " << programName << " --no-vad --no-low-power    # Full logging and recording\n";
#else
    std::cout << "  " << programName << "                           # Default: full features, WAV recording on\n";
    std::cout << "  " << programName << " --channels 8 --vad --csv   # Multi-channel with VAD and CSV\n";
    std::cout << "  " << programName << " --bypass                   # Test mode\n";
#endif
}

bool parseArguments(int argc, char* argv[], 
                   std::filesystem::path& outputDir,
                   int& bitDepth,
                   int& numChannels,
                   bool& bypassDenoise,
                   bool& enableVAD,
                   bool& lowPowerMode,
                   bool& enableWavWrite,
                   bool& enableCsvLog)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return false;
        }
        else if (arg == "--bypass") bypassDenoise = true;
        else if (arg == "--no-vad") enableVAD = false;
        else if (arg == "--vad") enableVAD = true;
        else if (arg == "--no-low-power") lowPowerMode = false;
        else if (arg == "--low-power") lowPowerMode = true;
        else if (arg == "--wav") enableWavWrite = true;
        else if (arg == "--csv") enableCsvLog = true;
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputDir = argv[++i];
        }
        else if ((arg == "-b" || arg == "--bitdepth") && i + 1 < argc) {
            bitDepth = std::stoi(argv[++i]);
            if (bitDepth != 16 && bitDepth != 24 && bitDepth != 32) {
                std::cerr << "Error: Bit depth must be 16, 24, or 32\n";
                return false;
            }
        }
        else if ((arg == "-c" || arg == "--channels") && i + 1 < argc) {
            numChannels = std::stoi(argv[++i]);
            if (numChannels < 1 || numChannels > NUM_CHANNELS_MAX) {
                std::cerr << "Error: Channels must be 1-" << NUM_CHANNELS_MAX << "\n";
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
        // Enable hardware denormal handling
        denormal_control::AutoDisable autoDisable;
        
        // Set up Ctrl+C handler
        std::signal(SIGINT, intHandler);

        // Default parameters (profile-dependent)
        std::filesystem::path outputDir = "test_output";
        int bitDepth = 16;
        int numChannels = NUM_CHANNELS_DEFAULT;
        bool bypassDenoise = false;
        bool enableVAD = ENABLE_VAD_DEFAULT;
        bool lowPowerMode = LOW_POWER_DEFAULT;
        bool enableWavWrite = !LOW_POWER_DEFAULT;  // WAV off in low power mode by default
        bool enableCsvLog = false;  // CSV logging off by default

        // Parse command line arguments
        if (!parseArguments(argc, argv, outputDir, bitDepth, numChannels, 
                          bypassDenoise, enableVAD, lowPowerMode, enableWavWrite, enableCsvLog)) {
            return 0;
        }

        // Initialize PortAudio
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
            return 1;
        }

        // Get default audio devices
        PaDeviceIndex inDev = Pa_GetDefaultInputDevice();
        PaDeviceIndex outDev = Pa_GetDefaultOutputDevice();
        
        if (inDev == paNoDevice || outDev == paNoDevice) {
            std::cerr << "Error: No audio devices found\n";
            Pa_Terminate();
            return 1;
        }

        const PaDeviceInfo* inInfo = Pa_GetDeviceInfo(inDev);
        const PaDeviceInfo* outInfo = Pa_GetDeviceInfo(outDev);

        // Validate channel count against device capabilities
        int maxChannels = std::min({inInfo->maxInputChannels, outInfo->maxOutputChannels, NUM_CHANNELS_MAX});
        if (numChannels > maxChannels) {
            std::cout << "Warning: Requested " << numChannels << " channels, using " << maxChannels << "\n";
            numChannels = maxChannels;
        }

        // Print configuration
        std::cout << "=== Audio Denoise (";
#ifdef BUILD_WEARABLE
        std::cout << "WEARABLE";
#elif defined(BUILD_EMBEDDED)
        std::cout << "EMBEDDED";
#else
        std::cout << "DESKTOP";
#endif
        std::cout << " profile) ===\n";
        std::cout << "Config: " << numChannels << "ch, " << SAMPLE_RATE << "Hz, " 
                 << CIRCULAR_BUFFER_FRAMES << " frame buffer (" << BUFFER_LATENCY_MS << "ms)\n";
        std::cout << "VAD: " << (enableVAD ? "on" : "off") 
                 << ", Low power: " << (lowPowerMode ? "on" : "off")
                 << ", WAV recording: " << (enableWavWrite ? "on" : "off")
                 << ", CSV logging: " << (enableCsvLog ? "on" : "off");
        std::cout << "\n\n";

#if ENABLE_WAV_WRITING
        if (enableWavWrite && !std::filesystem::exists(outputDir)) {
            std::filesystem::create_directories(outputDir);
        }
#endif

        size_t bufferSize = CIRCULAR_BUFFER_FRAMES * static_cast<size_t>(numChannels);
        AudioIOContext audioIOCtx(bufferSize, numChannels, bypassDenoise, enableVAD, lowPowerMode, enableWavWrite, enableCsvLog);

#if ENABLE_WAV_WRITING
        if (enableWavWrite) {
            auto inputPath  = outputDir / "input_raw.wav";
            auto outputPath = outputDir / "output_denoised.wav";
            audioIOCtx.wavInput  = std::make_unique<WavWriter>(inputPath.string(), SAMPLE_RATE, numChannels, bitDepth);
            audioIOCtx.wavOutput = std::make_unique<WavWriter>(outputPath.string(), SAMPLE_RATE, numChannels, bitDepth);
        }
#endif

#if ENABLE_FILE_LOGGING
        if (enableCsvLog) {
            auto logPath = outputDir / "rms_log.csv";
            audioIOCtx.logFilename = logPath.string();
            audioIOCtx.logFile.open(logPath, std::ios::out);
            if (audioIOCtx.logFile.is_open()) {
                audioIOCtx.logFile << "frame,in_rms,out_rms,processed\n";
            }
        } else {
            auto logPath = outputDir / "rms_log.txt";
            audioIOCtx.logFilename = logPath.string();
            audioIOCtx.logFile.open(logPath, std::ios::out);
            if (audioIOCtx.logFile.is_open()) {
                audioIOCtx.logFile << "frame in_rms out_rms processed\n";
            }
        }
#endif

        std::cout << "Press Ctrl+C to stop...\n\n";

        PaStream* stream = nullptr;
        
        if (SAMPLE_RATE != 48000) {
            std::cerr << "ERROR: RNNoise requires 48kHz sample rate!\n";
            Pa_Terminate();
            return 1;
        }
        
        err = Pa_OpenDefaultStream(&stream, numChannels, numChannels, paFloat32,
                                   SAMPLE_RATE, FRAME_SIZE, audioCallback, &audioIOCtx);
        
        if (err != paNoError) {
            std::cerr << "Error opening stream: " << Pa_GetErrorText(err) << std::endl;
            Pa_Terminate();
            return 1;
        }

        err = Pa_StartStream(stream);
        if (err != paNoError) {
            std::cerr << "Error starting stream: " << Pa_GetErrorText(err) << std::endl;
            Pa_CloseStream(stream);
            Pa_Terminate();
            return 1;
        }

        std::vector<float> silence(FRAME_SIZE * numChannels * 2, 0.0f);
        audioIOCtx.outputBuffer.pushBulk(silence.data(), silence.size());

        std::thread procThread(processingThread, &audioIOCtx, numChannels);

        while (keepRunning) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "\nShutting down...\n";
        Pa_StopStream(stream);
        procThread.join();
        Pa_CloseStream(stream);
        Pa_Terminate();

#if ENABLE_FILE_LOGGING
        audioIOCtx.logFile.close();
        if (!enableCsvLog) {
            audioIOCtx.convertLogToCSV(outputDir);
        }
#endif

#if ENABLE_WAV_WRITING
        if (enableWavWrite) {
            auto inputPath  = outputDir / "input_raw.wav";
            auto outputPath = outputDir / "output_denoised.wav";
            std::cout << "Files saved:\n";
            std::cout << "  Input:  " << inputPath << "\n";
            std::cout << "  Output: " << outputPath << "\n";
#if ENABLE_FILE_LOGGING
            if (enableCsvLog) {
                std::cout << "  Log:    " << outputDir / "rms_log.csv" << " (CSV)\n";
            } else {
                std::cout << "  Log:    " << outputDir / "rms_log.txt" << " (text)\n";
                std::cout << "  Log:    " << outputDir / "rms_log.csv" << " (CSV)\n";
            }
#endif
        }
#endif

        std::cout << "Done.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        Pa_Terminate();
        return 1;
    }
}