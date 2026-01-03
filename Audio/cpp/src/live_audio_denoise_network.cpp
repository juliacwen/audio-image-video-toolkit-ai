/*
 * @file live_audio_denoise_network.cpp
 * @brief Real-time audio denoising with RTP network streaming and adaptive jitter buffering
 * @author Julia Wen (wendigilane@gmail.com)
 * 
 * OVERVIEW:
 * A low-latency audio processing application that combines ML-based denoising with
 * network streaming capabilities for real-time audio transmission.
 * 
 * FEATURES:
 * - RNNoise ML-based noise suppression
 * - Voice Activity Detection (VAD)
 * - RTP network streaming (sender/receiver modes)
 * - Adaptive jitter buffer (40-200ms) with packet loss handling
 * - Optional WAV recording
 * - Test tone generation (440Hz)
 * 
 * ARCHITECTURE:
 * - PortAudio callback thread: Audio I/O with lock-free ring buffers
 * - Processing thread: Denoise, VAD, network send
 * - Network thread (receiver): UDP receive, jitter buffer management
 * 
 * MODES:
 * - Local: Mic → Denoise → Speaker (works on all platforms)
 * - Sender: Mic → Denoise → RTP → Network (Linux/macOS only)
 * - Receiver: Network → RTP → Jitter Buffer → Speaker (Linux/macOS only)
 * 
 * JITTER BUFFER (Step 2 - Implemented):
 * - Adaptive buffering: 40-200ms based on network conditions
 * - Packet loss handling: Inserts silence for lost packets
 * - Reordering: Handles out-of-order packets
 * - Statistics: Tracks packets received, lost, duplicates, underruns, overflows
 * - Underrun protection: Rebuffers if the buffer empties
 * 
 * How it works:
 * - Packets arrive → Added to jitter buffer (sorted by sequence)
 * - Processing thread pulls samples at steady rate
 * - Buffer starts playing once it reaches target latency (80ms default)
 * - Adapts buffer size up if underruns occur, down if consistently full
 * 
 * USAGE:
 *   Local:    ./live_audio_denoise_network
 *   Sender:   ./live_audio_denoise_network --send --dest 192.168.1.100 --port 5004
 *   Test:     ./live_audio_denoise_network --send --test-tone --dest 127.0.0.1
 *   Receiver: ./live_audio_denoise_network --receive --port 5004
 * 
 * PLATFORM SUPPORT:
 *   ✓ Linux   - Full support (audio + network)
 *   ✓ macOS   - Full support (audio + network)
 *   ⚠ Windows - Audio only (network features disabled, local mode works)
 * 
 * DEPENDENCIES:
 *   - PortAudio: Cross-platform audio I/O
 *   - RNNoise: ML-based noise suppression
 *   - POSIX sockets: Network streaming (Linux/macOS)
 * 
 * @par Revision History
 * - 12-07-2025 — Step 1: Basic RTP send/receive added
 * - 12-09-2025 — Fixed WAV writer initialization bug
 * - 12-17-2025 — Update ctx pointer/object access, thread-safety (NetworkStats, WAV writer, alignment, allocations)
 * - 12-20-2025 — Fix output wav name and atomic operations
 * - 12-24-2025 — Step 2: Add adaptive jitter buffer with packet loss handling
 * - 01-02-2026 — Refactor: Extract network code to network_rtp.h/.cpp, AudioIOContext to separate files
 * 
 */

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <csignal>
#include <atomic>
#include <thread>
#include <vector>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <mutex>
#include "../inc/denoise_config.h"
#include "../inc/SPSCFloatBuffer.h"
#include "../inc/denormal_control.h"
#include "../inc/audio_jitter_buffer.h"
#include "../inc/audio_io_context.h"
#include "../inc/network_rtp.h"
#include "rnnoise.h"
#include "portaudio.h"

#if ENABLE_WAV_WRITING
#include "../inc/wav_writer.h"
#endif

// ------------------ Signal Handling ------------------
std::atomic<bool> keepRunning{true};
void intHandler(int) { keepRunning.store(false); }

// ------------------ PortAudio Callback ------------------
static int audioCallback(const void* inputBuffer, void* outputBuffer,
                         unsigned long framesPerBuffer,
                         const PaStreamCallbackTimeInfo*,
                         PaStreamCallbackFlags,
                         void* userData) noexcept
{
    auto* ctx = static_cast<AudioIOContext*>(userData);
    const auto* in = static_cast<const float*>(inputBuffer);
    auto* out = static_cast<float*>(outputBuffer);
    
    const size_t totalSamples = framesPerBuffer * ctx->numChannels;
    
    // Input (only if we have input channels)
    if (in) {
        ctx->inputBuffer.pushBulk(in, totalSamples);
    }
    
    // Output (only if we have output channels)
    if (out) {
        size_t got = ctx->outputBuffer.popBulk(out, totalSamples);
        if (got < totalSamples) {
            std::fill(out + got, out + totalSamples, 0.0f);
        }
    }
    
    return paContinue;
}

// ------------------ Fast RMS Calculation ------------------
inline float calculateRMS(const float* data, size_t count, int stride = 1) {
    float sum = 0.0f;
    size_t i = 0, count4 = (count / 4) * 4;
    for (; i < count4; i += 4) {
        float s0 = data[i * stride], s1 = data[(i+1) * stride];
        float s2 = data[(i+2) * stride], s3 = data[(i+3) * stride];
        sum += s0*s0 + s1*s1 + s2*s2 + s3*s3;
    }
    for (; i < count; ++i) { 
        float s = data[i * stride]; 
        sum += s * s; 
    }
    return std::sqrt(sum / count);
}

// ------------------ Processing Thread ------------------
void processingThread(AudioIOContext* ctx, int numChannels)
{
    denormal_control::AutoDisable autoDisable;
    
    std::vector<float> inFrame(FRAME_SIZE * numChannels, 0.0f);
    std::vector<float> outFrame(FRAME_SIZE * numChannels, 0.0f);
    std::vector<float> inCh(FRAME_SIZE), outCh(FRAME_SIZE);
    
    const size_t totalSamplesNeeded = FRAME_SIZE * numChannels;
    size_t framesProcessed = 0;
    auto lastConsole = std::chrono::steady_clock::now();
    auto lastNetStats = lastConsole;
    
    // Target output buffer level (keep it low for low latency)
    const size_t TARGET_OUTPUT_BUFFER_SAMPLES = FRAME_SIZE * numChannels * 4;  // ~40ms buffer
    
    std::cout << "[Processing] Thread started\n";
    
    while (keepRunning) {
        // RECEIVER MODE: Get samples from jitter buffer
        if (ctx->networkReceive && !ctx->networkSend) {
            // Check output buffer level
            size_t outputAvail = ctx->outputBuffer.available();
            size_t outputCap = ctx->outputBuffer.capacity();
            size_t outputSpace = outputCap - outputAvail;
            
            // Check if it's time to print stats
            auto now = std::chrono::steady_clock::now();
            bool shouldPrintStats = std::chrono::duration_cast<std::chrono::seconds>(
                now - lastNetStats).count() >= CONSOLE_INTERVAL_SEC;
            
            // Only pull from jitter buffer if output buffer is getting low
            if (outputAvail > TARGET_OUTPUT_BUFFER_SAMPLES) {
                // Output buffer has enough data, don't pull more yet
                
                // Print Waiting stats only at interval
                if (shouldPrintStats && ctx->jitterBuffer) {
                    auto stats = ctx->jitterBuffer->getStats();
                    int latencyMs = ctx->jitterBuffer->getCurrentLatencyMs();
                    size_t buffered = ctx->jitterBuffer->getBufferedSamples();
                    
                    int outputLatencyMs = (outputAvail * MS_PER_SECOND) / (SAMPLE_RATE * numChannels);
                    int totalLatencyMs = latencyMs + outputLatencyMs;
                    
                    std::cout << "[Receiver:Waiting] Frames=" << framesProcessed
                              << ", Pkts=" << stats.packetsReceived
                              << ", Lost=" << stats.packetsLost
                              << ", TotalLatency=" << totalLatencyMs << "ms"
                              << " (Jitter=" << latencyMs << "ms + Output=" << outputLatencyMs << "ms)"
                              << ", JitterBuf=" << buffered << " samples"
                              << ", Underruns=" << stats.underruns << "\n";
                    lastNetStats = now;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            
            // Output buffer needs more data, check if we have space
            if (outputSpace < totalSamplesNeeded) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            
            if (ctx->jitterBuffer) {
                // Get samples from jitter buffer
                size_t got = ctx->jitterBuffer->getSamples(outFrame.data(), totalSamplesNeeded);
                
                // Push to output
                size_t pushed = ctx->outputBuffer.pushBulk(outFrame.data(), got);
                
                if (pushed < got) {
                    std::cerr << "[Receiver] Warning: Output buffer full, dropped " 
                              << (got - pushed) << " samples\n";
                }
                
#if ENABLE_WAV_WRITING
                // Record to WAV
                if (ctx->enableWavWrite && ctx->wavOutput) {
                    std::lock_guard<std::mutex> lock(ctx->wavMutex);
                    for (size_t i = 0; i < FRAME_SIZE; ++i) {
                        ctx->wavOutput->writeFrame(&outFrame[i * numChannels], numChannels);
                    }
                }
#endif
                
                framesProcessed++;
            }
            
            // Periodic stats (only when actively processing)
            if (shouldPrintStats && ctx->jitterBuffer) {
                auto stats = ctx->jitterBuffer->getStats();
                int latencyMs = ctx->jitterBuffer->getCurrentLatencyMs();
                size_t buffered = ctx->jitterBuffer->getBufferedSamples();
                outputAvail = ctx->outputBuffer.available();
                
                int outputLatencyMs = (outputAvail * MS_PER_SECOND) / (SAMPLE_RATE * numChannels);
                int totalLatencyMs = latencyMs + outputLatencyMs;
                
                std::cout << "[Receiver:Active] Frames=" << framesProcessed
                          << ", Pkts=" << stats.packetsReceived
                          << ", Lost=" << stats.packetsLost
                          << ", TotalLatency=" << totalLatencyMs << "ms"
                          << " (Jitter=" << latencyMs << "ms + Output=" << outputLatencyMs << "ms)"
                          << ", JitterBuf=" << buffered << " samples"
                          << ", Underruns=" << stats.underruns << "\n";
                lastNetStats = now;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // SENDER/LOCAL MODE: Process from microphone
        while (ctx->inputBuffer.available() < totalSamplesNeeded) {
            if (!keepRunning) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
        }
        
        // Read frame
        size_t got = 0;
        for (size_t i = 0; i < totalSamplesNeeded; i++) {
            float s;
            if (ctx->inputBuffer.pop(s)) {
                inFrame[i] = denormal_control::guardDenormal(s, ctx->denormalGuard);
                got++;
            } else break;
        }
        if (got < totalSamplesNeeded) continue;
        
        // If test tone mode, replace input with generated tone
        if (ctx->generateTestTone) {
            const double freq = 440.0;  // A4 note
            const double amplitude = 0.3;
            const double phaseIncrement = 2.0 * M_PI * freq / SAMPLE_RATE;
            
            for (size_t i = 0; i < totalSamplesNeeded; i += numChannels) {
                float sample = amplitude * std::sin(ctx->testTonePhase);
                for (int ch = 0; ch < numChannels; ++ch) {
                    inFrame[i + ch] = sample;
                }
                ctx->testTonePhase += phaseIncrement;
                if (ctx->testTonePhase >= 2.0 * M_PI) {
                    ctx->testTonePhase -= 2.0 * M_PI;
                }
            }
        }
        
        ctx->denormalGuard = -ctx->denormalGuard;
        
        float inRMS = calculateRMS(inFrame.data(), FRAME_SIZE, numChannels);
        
        // VAD
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
        
        // Denoise
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
        
        // NETWORK SEND
        if (ctx->networkSend && ctx->networkRTP) {
            ctx->networkRTP->sendPacket(outFrame.data(), FRAME_SIZE, numChannels);
        }
        
        // Local output (only if NOT in send-only mode)
        // IMPORTANT: In sender mode, we don't write to output buffer to prevent echo
        if (!ctx->networkSend || ctx->networkReceive) {
            ctx->outputBuffer.pushBulk(outFrame.data(), totalSamplesNeeded);
        }
        
#if ENABLE_WAV_WRITING
        if (ctx->enableWavWrite) {
            std::lock_guard<std::mutex> lock(ctx->wavMutex);
            // Write input if we have it
            if (ctx->wavInput) {
                for (int i = 0; i < FRAME_SIZE; ++i) {
                    ctx->wavInput->writeFrame(&inFrame[i * numChannels], numChannels);
                }
            }
            // Write output if we have it
            if (ctx->wavOutput) {
                for (int i = 0; i < FRAME_SIZE; ++i) {
                    ctx->wavOutput->writeFrame(&outFrame[i * numChannels], numChannels);
                }
            }
        }
#endif
        
        framesProcessed++;
        
#if ENABLE_FILE_LOGGING
        if (framesProcessed % LOG_EVERY_N_FRAMES == 0) {
            float outRMS = calculateRMS(outFrame.data(), FRAME_SIZE, numChannels);
            if (ctx->logFile.is_open()) {
                ctx->logFile << framesProcessed << " " << inRMS << " " << outRMS 
                            << " " << processFrame << "\n";
            }
        }
#endif
        
        // Console output
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(
            now - lastConsole).count() >= CONSOLE_INTERVAL_SEC) {
            float outRMS = calculateRMS(outFrame.data(), FRAME_SIZE, numChannels);
            std::cout << "[Sender] Frame=" << framesProcessed 
                      << ", in_rms=" << std::fixed << std::setprecision(3) << inRMS
                      << ", out_rms=" << outRMS;
            if (ctx->networkRTP) {
                std::cout << ", tx_pkts=" << ctx->networkRTP->getStats().packetsSent.load();
            }
            if (ctx->enableVAD) {
                std::cout << ", vad=" << (ctx->isVoiceActive ? "active" : "idle");
            }
            std::cout << "\n";
            lastConsole = now;
        }
        
        // Network stats
        if (ctx->networkSend && ctx->networkRTP) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(
                now - lastNetStats).count() >= CONSOLE_INTERVAL_SEC) {
                std::cout << "[Network] Transmitted " 
                          << ctx->networkRTP->getStats().packetsSent.load() << " packets\n";
                lastNetStats = now;
            }
        }
    }
    
    std::cout << "[Processing] Thread stopped\n";
}

// ------------------ Helper Functions ------------------
void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -c N             Channels (default: 1)\n";
    std::cout << "  --bypass         Bypass denoising\n";
    std::cout << "  --vad            Enable VAD\n";
    std::cout << "  --no-vad         Disable VAD\n";
    std::cout << "  --wav            Enable WAV recording (disabled by default)\n";
    std::cout << "\n";
    std::cout << "Network Options:\n";
    std::cout << "  --send           Sender mode\n";
    std::cout << "  --receive        Receiver mode\n";
    std::cout << "  --dest IP        Destination IP (default: 127.0.0.1)\n";
    std::cout << "  --port N         Port (default: 5004)\n";
    std::cout << "\n";
    std::cout << "Test Options:\n";
    std::cout << "  --test-tone      Generate 440Hz test tone (sender only)\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << "                            # Local only\n";
    std::cout << "  " << programName << " --send --dest 192.168.1.2  # Send mic\n";
    std::cout << "  " << programName << " --send --test-tone         # Send test tone\n";
    std::cout << "  " << programName << " --receive --port 5004      # Receive\n";
    std::cout << "  " << programName << " --send --wav               # Send with WAV recording\n";
}

bool parseArguments(int argc, char* argv[],
                   int& numChannels, bool& bypass, bool& vad, bool& wav,
                   bool& netSend, bool& netRecv, std::string& ip, int& port,
                   bool& testTone) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { printUsage(argv[0]); return false; }
        else if (arg == "--bypass") bypass = true;
        else if (arg == "--vad") vad = true;
        else if (arg == "--no-vad") vad = false;
        else if (arg == "--wav") wav = true;
        else if (arg == "--send") netSend = true;
        else if (arg == "--receive") netRecv = true;
        else if (arg == "--test-tone") testTone = true;
        else if (arg == "-c" && i+1 < argc) numChannels = std::stoi(argv[++i]);
        else if (arg == "--dest" && i+1 < argc) ip = argv[++i];
        else if (arg == "--port" && i+1 < argc) port = std::stoi(argv[++i]);
        else { std::cerr << "Unknown: " << arg << "\n"; return false; }
    }
    return true;
}

int main(int argc, char* argv[]) {
    try {
        denormal_control::AutoDisable autoDisable;
        std::signal(SIGINT, intHandler);
        
        int numChannels = NUM_CHANNELS_DEFAULT;
        bool bypass = false, vad = ENABLE_VAD_DEFAULT, wav = false;  // WAV disabled by default
        bool netSend = false, netRecv = false;
        bool testTone = false;
        std::string ip = "127.0.0.1";
        int port = 5004;
        
        if (!parseArguments(argc, argv, numChannels, bypass, vad, wav,
                          netSend, netRecv, ip, port, testTone)) return 0;
        
        std::cout << "=== Live Audio Denoise + Network ===\n";
        std::cout << "Mode: ";
        if (netSend) {
            std::cout << "SENDER to " << ip << ":" << port;
            if (testTone) std::cout << " (440Hz test tone)";
        }
        else if (netRecv) std::cout << "RECEIVER on port " << port;
        else std::cout << "LOCAL";
        std::cout << "\n\n";
        
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            std::cerr << "[PortAudio] Init failed: " << Pa_GetErrorText(err) << "\n";
            return 1;
        }
        
        size_t bufSz = CIRCULAR_BUFFER_FRAMES * numChannels;
        AudioIOContext ctx(bufSz, numChannels, bypass, vad, false, wav);
        ctx.networkSend = netSend;
        ctx.networkReceive = netRecv;
        ctx.generateTestTone = testTone;
        
        if (testTone && !netSend) {
            std::cout << "[Warning] --test-tone only works in sender mode, ignoring\n";
            ctx.generateTestTone = false;
        }
        
        bool hasInput = (netSend && !netRecv) || (!netSend && !netRecv);
        bool hasOutput = true;
        
        if (ctx.enableWavWrite) {
            ctx.initWavWriters(hasInput, hasOutput, netSend, netRecv);
        }
        
        // Initialize network if needed
        if (netSend || netRecv) {
            ctx.networkRTP = std::make_unique<NetworkRTP>();
            if (!ctx.networkRTP->init(netSend, netRecv, ip, port, port)) {
                Pa_Terminate();
                return 1;
            }
        }
        
        std::thread* netThread = nullptr;
        if (netRecv && ctx.networkRTP) {
            netThread = new std::thread(&NetworkRTP::receiveThreadFunc, ctx.networkRTP.get(), &ctx);
        }
        
        PaStream* stream = nullptr;
        
        int inputChannels = 0;
        int outputChannels = 0;
        
        // Sender mode: Captures mic, no speaker output
        // Receiver mode: No mic input, plays on speaker
        // Local mode: Both enabled (for testing without network)
        if (netSend && !netRecv) {
            inputChannels = numChannels;
            outputChannels = 0;
        } else if (netRecv && !netSend) {
            inputChannels = 0;
            outputChannels = numChannels;
        } else {
            inputChannels = numChannels;
            outputChannels = 0;
        }
                
        err = Pa_OpenDefaultStream(&stream, inputChannels, outputChannels, paFloat32,
                                   SAMPLE_RATE, FRAME_SIZE, audioCallback, &ctx);
        if (err != paNoError) {
            std::cerr << "[PortAudio] Failed to open stream: " << Pa_GetErrorText(err) << "\n";
            if (netThread) { 
                keepRunning = false;
                netThread->join(); 
                delete netThread; 
            }
            Pa_Terminate();
            return 1;
        }
        
        std::cout << "[PortAudio] Stream opened: in=" << inputChannels 
                  << " out=" << outputChannels << " @ " << SAMPLE_RATE << "Hz\n";
        
        if (inputChannels > 0 && outputChannels > 0) {
            std::cout << "[WARNING] Both input and output enabled - Risk of acoustic feedback!\n";
            std::cout << "[WARNING] Use headphones or the sound will loop back into the microphone\n";
        }
        
        err = Pa_StartStream(stream);
        if (err != paNoError) {
            std::cerr << "[PortAudio] Failed to start stream: " << Pa_GetErrorText(err) << "\n";
            Pa_CloseStream(stream);
            if (netThread) { 
                keepRunning = false;
                netThread->join(); 
                delete netThread; 
            }
            Pa_Terminate();
            return 1;
        }
        
        std::thread procThread(processingThread, &ctx, numChannels);
        
        std::cout << "Running... Ctrl+C to stop\n\n";
        while (keepRunning) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        std::cout << "\nStopping...\n";
        Pa_StopStream(stream);
        procThread.join();
        if (netThread) { netThread->join(); delete netThread; }
        Pa_CloseStream(stream);
        Pa_Terminate();
        
        std::cout << "Done.\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        Pa_Terminate();
        return 1;
    }
}