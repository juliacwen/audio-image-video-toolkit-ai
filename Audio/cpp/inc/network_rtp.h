/**
 * @file network_rtp.h
 * @brief RTP network streaming for real-time audio over UDP
 * @author Julia Wen (wendigilane@gmail.com)
 * 
 * This module provides RTP (Real-time Transport Protocol) packet handling for
 * audio streaming. Features:
 * - UDP socket management (non-blocking)
 * - RTP packet serialization/deserialization
 * - Sequence numbering and timestamp management
 * - Network statistics tracking
 * - Receive thread with jitter buffer integration
 * 
 * Platform support:
 *   ✓ Linux   - POSIX sockets
 *   ✓ macOS   - POSIX sockets
 *   ✗ Windows - Not currently supported (would require Winsock2 implementation)
 * 
 * Network format: Raw float audio (32-bit) in RTP payload
 * 
 * Usage:
 *   Sender: NetworkRTP::sendPacket() - Called from processing thread
 *   Receiver: NetworkRTP::receiveThreadFunc() - Runs in separate thread
 * 
 * Note: On Windows, the application will compile but network features will be
 * disabled. Only local audio processing (mic -> denoise -> speaker) will work.
 * 
 * @par Revision History
 * - 01-02-2026 — Initial Checkin
 */

#pragma once

#ifndef NETWORK_RTP_H
#define NETWORK_RTP_H

#include <string>
#include <vector>
#include <atomic>
#include <memory>
#include <cstdint>
#include <cstddef>
#include <chrono>
#include <iostream>

// Forward declarations
class AudioJitterBuffer;
struct AudioIOContext;

// Network configuration
// Note: Windows would require Winsock2 instead of POSIX sockets
#ifdef __linux__
#define NETWORK_SUPPORTED 1
#elif defined(__APPLE__)
#define NETWORK_SUPPORTED 1
#else
#define NETWORK_SUPPORTED 0  // Windows and other platforms
#endif

// RTP Header structure
#pragma pack(push, 1)
struct RTPHeader {
    uint8_t vpxcc;           // version(2), padding(1), extension(1), csrc count(4)
    uint8_t mpt;             // marker(1), payload type(7)
    uint16_t sequenceNumber; // Network byte order
    uint32_t timestamp;      // Network byte order
    uint32_t ssrc;           // Network byte order
    
    // Constructor to initialize all fields
    RTPHeader() 
        : vpxcc(0x80)           // version 2, no padding, no extension, no CSRC
        , mpt(111)              // Payload type (Opus default, using for raw audio)
        , sequenceNumber(0)
        , timestamp(0)
        , ssrc(0)
    {}
};
#pragma pack(pop)

// RTP Packet structure
struct RTPPacket {
    RTPHeader header;
    std::vector<uint8_t> payload;
    std::chrono::steady_clock::time_point arrivalTime;
};

// Network statistics
struct NetworkStats {
    std::atomic<uint64_t> packetsSent{0};      
    std::atomic<uint64_t> packetsReceived{0};  
    std::atomic<uint32_t> packetsLost{0};
    
    void logStats() const {
        std::cout << "[Network] Sent=" << packetsSent.load() 
                  << ", Received=" << packetsReceived.load()
                  << ", Lost=" << packetsLost.load() << "\n";
    }
};

// Network configuration constants
constexpr size_t MAX_RTP_PACKET_SIZE = 4096;
constexpr size_t RTP_BUFFER_SIZE = 4096;

// Network manager class
class NetworkRTP {
public:
    NetworkRTP();
    ~NetworkRTP();
    
    // Initialize network socket
    bool init(bool isSender, bool isReceiver, 
              const std::string& destIP, int destPort, int listenPort);
    
    // Send RTP packet
    void sendPacket(const float* audioData, size_t samples, int channels);
    
    // Receive thread function (to be run in separate thread)
    // NOTE: Requires full AudioIOContext definition to be included
    void receiveThreadFunc(AudioIOContext* ctx);
    
    // Cleanup
    void shutdown();
    
    // Getters
    int getSocket() const { return m_socket; }
    const NetworkStats& getStats() const { return m_stats; }
    
    // Sequence and timestamp management
    uint16_t getNextSequence();
    uint32_t getCurrentTimestamp() const;
    void incrementTimestamp(uint32_t samples);

private:
    int m_socket;
    bool m_isSender;
    bool m_isReceiver;
    std::string m_destIP;
    int m_destPort;
    int m_listenPort;
    uint32_t m_ssrc;
    
    std::atomic<uint16_t> m_rtpSequence{0};
    std::atomic<uint32_t> m_rtpTimestamp{0};
    NetworkStats m_stats;
    
    // Pre-allocated RTP packet buffer
    std::vector<uint8_t> m_rtpSendBuffer;
};

#endif // NETWORK_RTP_H