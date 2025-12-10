/*
 * @file live_audio_denoise_network.cpp
 * @brief Live Audio Denoising with Network Streaming Support (Step 1: Basic RTP)
 * @author Julia Wen (wendigilane@gmail.com)
 * 
 * STEP 1: Basic RTP Send/Receive
 * - RTP packet structure and serialization
 * - UDP socket setup (Linux/macOS)
 * - Sender mode: Denoise → RTP → Network
 * - Receiver mode: Network → RTP → Speaker (direct, no jitter buffer yet)
 * 
 * USAGE:
 *  Local:     ./live_audio_denoise_network
 *  Sender:    ./live_audio_denoise_network --send --dest 192.168.1.100 --port 5004
 *  Receiver:  ./live_audio_denoise_network --receive --port 5004
 * 
 * @par Revision History
 * - 12-07-2025 — Step 1: Basic RTP send/receive added
 * - 12-09-2025 — Fixed WAV writer initialization bug
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
#include <cstring>
#include "../inc/denoise_config.h"
#include "../inc/SPSCFloatBuffer.h"
#include "../inc/denormal_control.h"
#include "rnnoise.h"
#include "portaudio.h"

#if ENABLE_WAV_WRITING
#include "../inc/wav_writer.h"
#endif

// Network includes (Linux/macOS only for now)
#ifdef __linux__
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#define NETWORK_SUPPORTED 1
#elif defined(__APPLE__)
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#define NETWORK_SUPPORTED 1
#else
#define NETWORK_SUPPORTED 0
#endif

// ------------------ Signal Handling ------------------
std::atomic<bool> keepRunning{true};
void intHandler(int) { keepRunning.store(false); }

// ------------------ RTP Structures ------------------

#pragma pack(push, 1)
struct RTPHeader {
    uint8_t vpxcc;           // version(2), padding(1), extension(1), csrc count(4)
    uint8_t mpt;             // marker(1), payload type(7)
    uint16_t sequenceNumber; // Network byte order
    uint32_t timestamp;      // Network byte order
    uint32_t ssrc;           // Network byte order
    
    RTPHeader() {
        vpxcc = 0x80;  // version 2, no padding, no extension, no CSRC
        mpt = 111;     // Opus payload type (we'll use for raw audio for now)
        sequenceNumber = 0;
        timestamp = 0;
        ssrc = 0;
    }
};
#pragma pack(pop)

struct RTPPacket {
    RTPHeader header;
    std::vector<uint8_t> payload;
    std::chrono::steady_clock::time_point arrivalTime;
};

// ------------------ Network Statistics ------------------
struct NetworkStats {
    uint32_t packetsSent = 0;
    uint32_t packetsReceived = 0;
    uint32_t packetsLost = 0;
    
    void logStats() const {
        std::cout << "[Network] Sent=" << packetsSent 
                  << ", Received=" << packetsReceived
                  << ", Lost=" << packetsLost << "\n";
    }
};

// ------------------ Audio IO Context (Extended) ------------------
struct AudioIOContext {
    // Original baseline members
    std::vector<DenoiseState*> states;
    SPSCFloatBuffer inputBuffer;
    SPSCFloatBuffer outputBuffer;

#if ENABLE_WAV_WRITING
    std::unique_ptr<WavWriter> wavInput;
    std::unique_ptr<WavWriter> wavOutput;
#endif

#if ENABLE_FILE_LOGGING
    std::ofstream logFile;
#endif
    
    bool bypassDenoise;
    bool enableVAD;
    bool lowPowerMode;
    bool enableWavWrite;
    float denormalGuard;
    int numChannels;
    
    int vadHangoverCounter;
    bool isVoiceActive;
    float smoothedRMS;
    
    // NEW: Network state
    bool networkSend = false;
    bool networkReceive = false;
    std::string destIP = "127.0.0.1";
    int destPort = 5004;
    int listenPort = 5004;
    int networkSocket = -1;
    uint16_t rtpSequence = 0;
    uint32_t rtpTimestamp = 0;
    uint32_t ssrc = 0x12345678;  // Random SSRC
    NetworkStats netStats;
    
    // Test tone generator
    bool generateTestTone = false;
    double testTonePhase = 0.0;

    AudioIOContext(size_t bufferSize, int numCh, bool bypass, bool vad, bool lowPower, bool wavWrite)
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
        
#if ENABLE_WAV_WRITING
        // Initialize WAV writers if enabled
        if (enableWavWrite) {
            try {
                wavInput = std::make_unique<WavWriter>("input.wav", SAMPLE_RATE, numCh);
                wavOutput = std::make_unique<WavWriter>("output.wav", SAMPLE_RATE, numCh);
                std::cout << "[WAV] Recording enabled: input.wav and output.wav\n";
            } catch (const std::exception& e) {
                std::cerr << "[WAV] Failed to initialize: " << e.what() << "\n";
                enableWavWrite = false;
            }
        }
#else
        if (enableWavWrite) {
            std::cout << "[WAV] WAV writing is disabled in this build\n";
            enableWavWrite = false;
        }
#endif

#if ENABLE_FILE_LOGGING
        logFile.open("denoise.log");
        if (logFile.is_open()) {
            logFile << "frame in_rms out_rms processed\n";
        }
#endif
    }

    ~AudioIOContext() {
        for (auto* state : states) {
            if (state) rnnoise_destroy(state);
        }
#if NETWORK_SUPPORTED
        if (networkSocket >= 0) {
            close(networkSocket);
        }
#endif
    }
};

// ------------------ Network Functions ------------------

#if NETWORK_SUPPORTED

bool initNetwork(AudioIOContext* ctx) {
    ctx->networkSocket = socket(AF_INET, SOCK_DGRAM, 0);
    if (ctx->networkSocket < 0) {
        std::cerr << "[Network] Failed to create UDP socket\n";
        return false;
    }
    
    // Set non-blocking
    int flags = fcntl(ctx->networkSocket, F_GETFL, 0);
    fcntl(ctx->networkSocket, F_SETFL, flags | O_NONBLOCK);
    
    // Bind for receiving
    if (ctx->networkReceive) {
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(ctx->listenPort);
        addr.sin_addr.s_addr = INADDR_ANY;
        
        if (bind(ctx->networkSocket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "[Network] Failed to bind to port " << ctx->listenPort << "\n";
            close(ctx->networkSocket);
            ctx->networkSocket = -1;
            return false;
        }
        
        std::cout << "[Network] Listening on port " << ctx->listenPort << "\n";
    }
    
    // For sender, just log destination
    if (ctx->networkSend) {
        std::cout << "[Network] Will send to " << ctx->destIP << ":" << ctx->destPort << "\n";
    }
    
    return true;
}

void sendRTPPacket(AudioIOContext* ctx, const float* audioData, size_t samples, int channels) {
    if (ctx->networkSocket < 0) return;
    
    RTPPacket packet;
    packet.header.sequenceNumber = htons(ctx->rtpSequence++);
    packet.header.timestamp = htonl(ctx->rtpTimestamp);
    packet.header.ssrc = htonl(ctx->ssrc);
    
    // Update timestamp (samples per frame, not per channel)
    ctx->rtpTimestamp += samples;
    
    // Payload: raw float audio (interleaved if multi-channel)
    // TODO: In production, encode with Opus here
    size_t totalSamples = samples * channels;
    size_t byteSize = totalSamples * sizeof(float);
    packet.payload.resize(byteSize);
    std::memcpy(packet.payload.data(), audioData, byteSize);
    
    // Send via UDP
    struct sockaddr_in dest;
    memset(&dest, 0, sizeof(dest));
    dest.sin_family = AF_INET;
    dest.sin_port = htons(ctx->destPort);
    if (inet_pton(AF_INET, ctx->destIP.c_str(), &dest.sin_addr) <= 0) {
        std::cerr << "[Network] Invalid destination IP: " << ctx->destIP << "\n";
        return;
    }
    
    // Serialize: header + payload
    std::vector<uint8_t> buffer(sizeof(RTPHeader) + byteSize);
    std::memcpy(buffer.data(), &packet.header, sizeof(RTPHeader));
    std::memcpy(buffer.data() + sizeof(RTPHeader), packet.payload.data(), byteSize);
    
    ssize_t sent = sendto(ctx->networkSocket, buffer.data(), buffer.size(), 0,
                         (struct sockaddr*)&dest, sizeof(dest));
    
    if (sent > 0) {
        ctx->netStats.packetsSent++;
    } else if (sent < 0) {
        // Non-blocking socket, so EAGAIN/EWOULDBLOCK is normal
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            std::cerr << "[Network] Send error: " << strerror(errno) << "\n";
        }
    }
}

// Network receive thread
void networkReceiveThread(AudioIOContext* ctx) {
    uint8_t buffer[2048];
    
    std::cout << "[Network] Receive thread started\n";
    
    uint32_t lastSeq = 0;
    bool firstPacket = true;
    
    while (keepRunning) {
        ssize_t received = recvfrom(ctx->networkSocket, buffer, sizeof(buffer), 
                                   0, nullptr, nullptr);
        
        if (received > static_cast<ssize_t>(sizeof(RTPHeader))) {
            // Parse RTP packet
            RTPPacket packet;
            std::memcpy(&packet.header, buffer, sizeof(RTPHeader));
            packet.header.sequenceNumber = ntohs(packet.header.sequenceNumber);
            packet.header.timestamp = ntohl(packet.header.timestamp);
            packet.header.ssrc = ntohl(packet.header.ssrc);
            
            // Detect packet loss
            if (!firstPacket) {
                uint32_t expected = (lastSeq + 1) & 0xFFFF;
                if (packet.header.sequenceNumber != expected) {
                    uint32_t lost = (packet.header.sequenceNumber - expected) & 0xFFFF;
                    ctx->netStats.packetsLost += lost;
                    std::cout << "[Network] Lost " << lost << " packets (seq " 
                              << expected << " to " << packet.header.sequenceNumber-1 << ")\n";
                }
            }
            lastSeq = packet.header.sequenceNumber;
            firstPacket = false;
            
            size_t payloadSize = received - sizeof(RTPHeader);
            packet.payload.assign(buffer + sizeof(RTPHeader), 
                                buffer + received);
            packet.arrivalTime = std::chrono::steady_clock::now();
            
            ctx->netStats.packetsReceived++;
            
            // Debug output for first few packets
            if (ctx->netStats.packetsReceived <= 5) {
                std::cout << "[Network] Received packet #" << ctx->netStats.packetsReceived
                          << " seq=" << packet.header.sequenceNumber
                          << " size=" << payloadSize << " bytes\n";
            }
            
            // STEP 1: Direct playback (no jitter buffer yet)
            // Copy payload directly to output buffer
            size_t samples = payloadSize / sizeof(float);
            if (samples > 0) {
                size_t pushed = ctx->outputBuffer.pushBulk((const float*)packet.payload.data(), samples);
                if (pushed < samples) {
                    std::cout << "[Network] Output buffer full, dropped " << (samples - pushed) << " samples\n";
                }
            }
        } else if (received < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            std::cerr << "[Network] Receive error: " << strerror(errno) << "\n";
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    std::cout << "[Network] Receive thread stopped\n";
}

#else // !NETWORK_SUPPORTED

bool initNetwork(AudioIOContext*) {
    std::cerr << "[Network] Network streaming not supported on this platform\n";
    return false;
}

void sendRTPPacket(AudioIOContext*, const float*, size_t, int) {}
void networkReceiveThread(AudioIOContext*) {}

#endif // NETWORK_SUPPORTED

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
    
    std::cout << "[Processing] Thread started\n";
    
    while (keepRunning) {
        // RECEIVER MODE: Just pass through network audio (for now)
        if (ctx->networkReceive && !ctx->networkSend) {
            // Network thread handles receiving and pushing to output buffer
            // Just log stats periodically
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(
                now - lastNetStats).count() >= 5) {
                
                // Check output buffer status
                size_t available = ctx->outputBuffer.available();
                std::cout << "[Receiver] Packets=" << ctx->netStats.packetsReceived
                          << ", Lost=" << ctx->netStats.packetsLost
                          << ", OutputBuf=" << available << " samples\n";
                lastNetStats = now;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
        if (ctx->networkSend) {
            sendRTPPacket(ctx, outFrame.data(), FRAME_SIZE, numChannels);
        }
        
        // Local output (only if NOT in send-only mode)
        if (!ctx->networkSend || ctx->networkReceive) {
            ctx->outputBuffer.pushBulk(outFrame.data(), totalSamplesNeeded);
        }
        
#if ENABLE_WAV_WRITING
        if (ctx->enableWavWrite && ctx->wavInput && ctx->wavOutput) {
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
            std::cout << "[Frame " << framesProcessed << "] in_rms=" << inRMS
                      << ", out_rms=" << outRMS;
            if (ctx->enableVAD) {
                std::cout << ", vad=" << (ctx->isVoiceActive ? "active" : "idle");
            }
            if (ctx->networkSend) {
                std::cout << ", net_tx=" << ctx->netStats.packetsSent;
            }
            std::cout << "\n";
            lastConsole = now;
        }
        
        // Network stats
        if (ctx->networkSend) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(
                now - lastNetStats).count() >= 10) {
                ctx->netStats.logStats();
                lastNetStats = now;
            }
        }
    }
    
    std::cout << "[Processing] Thread stopped\n";
}

// ------------------ Helper Functions ------------------
void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n\n";
    std::cout << "STEP 1: Basic RTP send/receive (no jitter buffer yet)\n\n";
    std::cout << "Options:\n";
    std::cout << "  -c N             Channels (default: 1)\n";
    std::cout << "  --bypass         Bypass denoising\n";
    std::cout << "  --vad            Enable VAD\n";
    std::cout << "  --no-vad         Disable VAD\n";
    std::cout << "  --wav            Enable WAV recording\n";
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

// ------------------ Main ------------------
int main(int argc, char* argv[]) {
    try {
        denormal_control::AutoDisable autoDisable;
        std::signal(SIGINT, intHandler);
        
        int numChannels = NUM_CHANNELS_DEFAULT;
        bool bypass = false, vad = ENABLE_VAD_DEFAULT, wav = !LOW_POWER_DEFAULT;
        bool netSend = false, netRecv = false;
        bool testTone = false;
        std::string ip = "127.0.0.1";
        int port = 5004;
        
        if (!parseArguments(argc, argv, numChannels, bypass, vad, wav,
                          netSend, netRecv, ip, port, testTone)) return 0;
        
        std::cout << "=== Live Audio Denoise + Network (Step 1) ===\n";
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
        ctx.destIP = ip;
        ctx.destPort = port;
        ctx.listenPort = port;
        ctx.generateTestTone = testTone;
        
        if (testTone && !netSend) {
            std::cout << "[Warning] --test-tone only works in sender mode\n";
            ctx.generateTestTone = false;
        }
        
        if (netSend || netRecv) {
            if (!initNetwork(&ctx)) {
                Pa_Terminate();
                return 1;
            }
        }
        
        std::thread* netThread = nullptr;
        if (netRecv) {
            netThread = new std::thread(networkReceiveThread, &ctx);
        }
        
        PaStream* stream = nullptr;
        
        // Configure audio channels based on mode:
        // - Send-only: Need input (mic), NO output (muted)
        // - Receive-only: NO input, need output (speakers)
        // - Local/both: Need both input and output
        int inputChannels = 0;
        int outputChannels = 0;
        
        if (netSend && !netRecv) {
            // Send-only: mic input, no local playback
            inputChannels = numChannels;
            outputChannels = 0;
        } else if (netRecv && !netSend) {
            // Receive-only: no mic, only speakers
            inputChannels = 0;
            outputChannels = numChannels;
        } else {
            // Local mode or simultaneous send/receive
            inputChannels = numChannels;
            outputChannels = numChannels;
        }
        
        err = Pa_OpenDefaultStream(&stream, inputChannels, outputChannels, paFloat32,
                                   SAMPLE_RATE, FRAME_SIZE, audioCallback, &ctx);
        if (err != paNoError) {
            std::cerr << "[PortAudio] Failed to open stream: " << Pa_GetErrorText(err) << "\n";
            std::cerr << "[PortAudio] Channels: in=" << inputChannels << ", out=" << outputChannels << "\n";
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
        if (netSend && !netRecv) {
            std::cout << "[Mode] Send-only - local playback DISABLED\n";
            if (wav) {
                std::cout << "[Mode] WAV recording will capture input and denoised output\n";
            }
        } else if (netRecv && !netSend) {
            std::cout << "[Mode] Receive-only - microphone DISABLED\n";
            if (wav) {
                std::cout << "[Mode] WARNING: WAV recording disabled in receive-only mode\n";
                ctx.enableWavWrite = false;  // Disable WAV in receive mode
            }
        } else if (!netSend && !netRecv) {
            std::cout << "[Mode] Local processing only\n";
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