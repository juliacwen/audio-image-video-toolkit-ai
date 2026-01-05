/**
 * @file network_rtp.cpp
 * @brief Implementation of RTP network streaming
 * @author Julia Wen (wendigilane@gmail.com)
 * 
 * Handles all network operations for audio streaming:
 * - Socket creation and configuration (UDP, non-blocking)
 * - RTP packet transmission with atomic sequence/timestamp management
 * - Network receive thread with packet validation
 * - Integration with AudioJitterBuffer for playout
 * - Error handling and statistics reporting
 * 
 * Threading model:
 * - sendPacket(): Called from processing thread (safe via atomics)
 * - receiveThreadFunc(): Runs in dedicated network thread
 * 
 * Platform implementation:
 * - Linux/macOS: POSIX sockets (sys/socket.h, netinet/in.h, arpa/inet.h)
 * - Windows: Not implemented (would need Winsock2: winsock2.h, ws2tcpip.h)
 * 
 * To add Windows support, would need to:
 * 1. Replace POSIX socket calls with Winsock2 equivalents
 * 2. Call WSAStartup()/WSACleanup()
 * 3. Use ioctlsocket() instead of fcntl() for non-blocking mode
 * 4. Handle WSAEWOULDBLOCK instead of EAGAIN/EWOULDBLOCK
 * 
 * @par Revision History
 * - 01-02-2026 — Initial Checkin
 * - 01-05-2026 — Add profile logging
 */

#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>
#include "../inc/network_rtp.h"
#include "../inc/audio_io_context.h"
#include "../inc/denoise_config.h"
#include "../inc/audio_jitter_buffer.h"

#if NETWORK_SUPPORTED
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#endif

// External keepRunning from main.cpp
extern std::atomic<bool> keepRunning;

NetworkRTP::NetworkRTP()
    : m_socket(-1)
    , m_isSender(false)
    , m_isReceiver(false)
    , m_destPort(5004)
    , m_listenPort(5004)
    , m_ssrc(0x12345678)  // Random SSRC
{
    m_rtpSendBuffer.reserve(RTP_BUFFER_SIZE);
}

NetworkRTP::~NetworkRTP() {
    shutdown();
}

bool NetworkRTP::init(bool isSender, bool isReceiver, 
                      const std::string& destIP, int destPort, int listenPort) {
#if !NETWORK_SUPPORTED
    std::cerr << "[Network] Network streaming not supported on this platform\n";
    return false;
#else
    m_isSender = isSender;
    m_isReceiver = isReceiver;
    m_destIP = destIP;
    m_destPort = destPort;
    m_listenPort = listenPort;
    
    m_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (m_socket < 0) {
        std::cerr << "[Network] Failed to create UDP socket\n";
        return false;
    }
    
    // Set non-blocking
    int flags = fcntl(m_socket, F_GETFL, 0);
    fcntl(m_socket, F_SETFL, flags | O_NONBLOCK);
    
    // Bind for receiving
    if (m_isReceiver) {
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(m_listenPort);
        addr.sin_addr.s_addr = INADDR_ANY;
        
        if (bind(m_socket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "[Network] Failed to bind to port " << m_listenPort << "\n";
            close(m_socket);
            m_socket = -1;
            return false;
        }
        
        std::cout << "[Network] Listening on port " << m_listenPort << "\n";
    }
    
    // For sender, just log destination
    if (m_isSender) {
        std::cout << "[Network] Will send to " << m_destIP << ":" << m_destPort << "\n";
    }
    
    return true;
#endif
}

void NetworkRTP::sendPacket(const float* audioData, size_t samples, int channels) {
#if !NETWORK_SUPPORTED
    return;
#else
    if (m_socket < 0) return;
    
    RTPHeader header;
    
    // ATOMIC OPERATION: fetch_add returns old value, then increments
    uint16_t seq = m_rtpSequence.fetch_add(1, std::memory_order_relaxed);
    header.sequenceNumber = htons(seq);
    
    // ATOMIC OPERATION: load current timestamp
    uint32_t ts = m_rtpTimestamp.load(std::memory_order_relaxed);
    header.timestamp = htonl(ts);
    
    // ATOMIC OPERATION: increment timestamp for next packet
    m_rtpTimestamp.fetch_add(samples, std::memory_order_relaxed);
    
    header.ssrc = htonl(m_ssrc);
    
    // Payload: raw float audio (interleaved if multi-channel)
    size_t totalSamples = samples * channels;
    size_t byteSize = totalSamples * sizeof(float);
    
    // Use pre-allocated buffer
    size_t packetSize = sizeof(RTPHeader) + byteSize;
    m_rtpSendBuffer.resize(packetSize);
    
    // Serialize: header + payload
    std::memcpy(m_rtpSendBuffer.data(), &header, sizeof(RTPHeader));
    std::memcpy(m_rtpSendBuffer.data() + sizeof(RTPHeader), audioData, byteSize);
    
    // Send via UDP
    struct sockaddr_in dest;
    memset(&dest, 0, sizeof(dest));
    dest.sin_family = AF_INET;
    dest.sin_port = htons(m_destPort);
    if (inet_pton(AF_INET, m_destIP.c_str(), &dest.sin_addr) <= 0) {
        std::cerr << "[Network] Invalid destination IP: " << m_destIP << "\n";
        return;
    }
    
    ssize_t sent = sendto(m_socket, m_rtpSendBuffer.data(), 
                         m_rtpSendBuffer.size(), 0,
                         (struct sockaddr*)&dest, sizeof(dest));
    
    if (sent > 0) {
        m_stats.packetsSent.fetch_add(1, std::memory_order_relaxed);
    } else if (sent < 0) {
        // Non-blocking socket, so EAGAIN/EWOULDBLOCK is normal
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            std::cerr << "[Network] Send error: " << strerror(errno) << "\n";
        }
    }
#endif
}

void NetworkRTP::receiveThreadFunc(AudioIOContext* ctx) {
#if !NETWORK_SUPPORTED
    std::cerr << "[Network] Receive not supported on this platform\n";
    return;
#else
    uint8_t buffer[MAX_RTP_PACKET_SIZE] = {0};
    
    std::cout << "[Network] Receive thread started\n";
    
    // Create jitter buffer - uses profile-specific JITTER_BUFFER_TARGET_MS from header
    std::cout << "[Network] Jitter buffer config: " << JITTER_BUFFER_TARGET_MS << "ms target, "
              << JITTER_BUFFER_MIN_MS << "-" << JITTER_BUFFER_MAX_MS << "ms range";
#ifdef BUILD_EMBEDDED
    std::cout << " (EMBEDDED profile)";
#elif defined(BUILD_WEARABLE)
    std::cout << " (WEARABLE profile)";
#else
    std::cout << " (DESKTOP profile)";
#endif
    std::cout << "\n";
    
    ctx->jitterBuffer = std::make_unique<AudioJitterBuffer>(
        SAMPLE_RATE, ctx->numChannels, JITTER_BUFFER_TARGET_MS);
    
    uint64_t receiveErrors = 0;
    uint64_t lastPacketCount = 0;
    auto lastErrorLog = std::chrono::steady_clock::now();
    auto lastActivityCheck = std::chrono::steady_clock::now();
    
    while (keepRunning) {
        ssize_t received = recvfrom(m_socket, buffer, sizeof(buffer), 
                                   0, nullptr, nullptr);
        
        if (received > static_cast<ssize_t>(sizeof(RTPHeader))) {
            // Parse RTP packet
            RTPPacket packet;
            std::memcpy(&packet.header, buffer, sizeof(RTPHeader));
            packet.header.sequenceNumber = ntohs(packet.header.sequenceNumber);
            packet.header.timestamp = ntohl(packet.header.timestamp);
            packet.header.ssrc = ntohl(packet.header.ssrc);
            
            size_t payloadSize = received - sizeof(RTPHeader);
            packet.payload.assign(buffer + sizeof(RTPHeader), 
                                buffer + received);
            packet.arrivalTime = std::chrono::steady_clock::now();
            
            // Validate payload size
            if (payloadSize % sizeof(float) != 0) {
                std::cerr << "[Network] Invalid payload size " << payloadSize << "\n";
                continue;
            }
            
            m_stats.packetsReceived.fetch_add(1, std::memory_order_relaxed);
            
            // Add to jitter buffer
            if (!ctx->jitterBuffer->addPacket(packet)) {
                // Packet rejected (duplicate or other issue)
                // This is normal, don't log
            }
            
            // Debug output for first few packets
            uint64_t pktCount = m_stats.packetsReceived.load();
            if (pktCount <= 5) {
                std::cout << "[Network] Received packet #" << pktCount
                          << " seq=" << packet.header.sequenceNumber
                          << " ssrc=0x" << std::hex << packet.header.ssrc << std::dec
                          << " size=" << payloadSize << " bytes"
                          << " buffered=" << ctx->jitterBuffer->getBufferedSamples() << " samples\n";
            }
            
        } else if (received < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                receiveErrors++;
                
                // Log errors but not too frequently (once per second max)
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(
                    now - lastErrorLog).count() >= 1) {
                    std::cerr << "[Network] Receive error: " << strerror(errno) 
                              << " (total errors: " << receiveErrors << ")\n";
                    lastErrorLog = now;
                }
            }
        } else if (received == 0) {
            std::cerr << "[Network] Socket closed\n";
            break;
        }
        
        // Check for inactivity (no packets for 5 seconds)
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(
            now - lastActivityCheck).count() >= 5) {
            
            uint64_t currentPacketCount = m_stats.packetsReceived.load();
            if (currentPacketCount == lastPacketCount) {
                std::cout << "[Network] WARNING: No packets received for 5 seconds "
                          << "(total: " << currentPacketCount << ", errors: " << receiveErrors << ")\n";
            }
            lastPacketCount = currentPacketCount;
            lastActivityCheck = now;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    std::cout << "[Network] Receive thread stopped\n";
    std::cout << "[Network] Total receive errors: " << receiveErrors << "\n";
    
    // Print final jitter buffer statistics
    if (ctx->jitterBuffer) {
        auto stats = ctx->jitterBuffer->getStats();
        std::cout << "[AudioJitterBuffer] Final stats:\n"
                  << "  Received: " << stats.packetsReceived << "\n"
                  << "  Lost: " << stats.packetsLost << "\n"
                  << "  Duplicates: " << stats.duplicates << "\n"
                  << "  Underruns: " << stats.underruns << "\n"
                  << "  Overflows: " << stats.overflow << "\n";
    }
#endif
}

void NetworkRTP::shutdown() {
#if NETWORK_SUPPORTED
    if (m_socket >= 0) {
        close(m_socket);
        m_socket = -1;
    }
#endif
}

uint16_t NetworkRTP::getNextSequence() {
    return m_rtpSequence.fetch_add(1, std::memory_order_relaxed);
}

uint32_t NetworkRTP::getCurrentTimestamp() const {
    return m_rtpTimestamp.load(std::memory_order_relaxed);
}

void NetworkRTP::incrementTimestamp(uint32_t samples) {
    m_rtpTimestamp.fetch_add(samples, std::memory_order_relaxed);
}