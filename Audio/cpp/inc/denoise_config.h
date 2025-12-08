#pragma once
/**
 * @file denoise_config.h
 * @brief live audio denoise with PortAudio and RNNoise build profile configuration.
 * This file defines compile-time constants based on the build profile
 * Build profile is set via CMake: -DBUILD_PROFILE=DESKTOP|WEARABLE|EMBEDDED
 * or via compile definitions: -DBUILD_DESKTOP, -DBUILD_WEARABLE, -DBUILD_EMBEDDED
 * @author Julia Wen (wendigilane@gmail.com)
 * @date 2025-12-07
 */

#ifndef DENOISE_CONFIG_H
#define DENOISE_CONFIG_H

// Default to DESKTOP if nothing specified
#if !defined(BUILD_WEARABLE) && !defined(BUILD_DESKTOP) && !defined(BUILD_EMBEDDED)
    #define BUILD_DESKTOP
#endif

// ------------------ Profile-Specific Configurations ------------------

#ifdef BUILD_WEARABLE
    // ============================================================
    // WEARABLE PROFILE: Battery-powered devices
    // ============================================================
    // Target: Phones, earbuds, AR/VR/XR headsets, smart glasses
    // Focus: Balance quality with battery life
    
    constexpr int SAMPLE_RATE = 48000;                // 48kHz (RNNoise requirement)
    constexpr int FRAME_SIZE = 480;                   // 10ms frames
    constexpr int NUM_CHANNELS_MAX = 8;               // AR/VR multi-mic arrays
    constexpr int CIRCULAR_BUFFER_FRAMES = 9600;      // 200ms buffer (lower latency)
    constexpr int CONSOLE_INTERVAL_SEC = 30;          // Less frequent console output
    constexpr int POLL_INTERVAL_MS = 2;               
    constexpr bool ENABLE_VAD_DEFAULT = true;         // VAD for power saving
    constexpr bool LOW_POWER_DEFAULT = true;          // Reduce I/O operations
    constexpr int LOG_EVERY_N_FRAMES = 100;           // Log every 1 second
    constexpr float VAD_THRESHOLD = 0.001f;           // Voice activity threshold
    constexpr int VAD_HANGOVER_FRAMES = 20;           // 200ms hangover
    
    #define ENABLE_WAV_WRITING 1
    #define ENABLE_FILE_LOGGING 1
    #define ENABLE_CONSOLE_OUTPUT 1
    
    #define PROFILE_NAME "WEARABLE"

#elif defined(BUILD_EMBEDDED)
    // ============================================================
    // EMBEDDED PROFILE: Minimal resources
    // ============================================================
    // Target: MCUs, IoT devices, smart speakers
    // Focus: Minimal memory and CPU usage
    
    constexpr int SAMPLE_RATE = 48000;                // 48kHz required by RNNoise (16kHz is intended for embedded)
    constexpr int FRAME_SIZE = 480;                   // 10ms frames
    constexpr int NUM_CHANNELS_MAX = 1;               // Mono only
    constexpr int CIRCULAR_BUFFER_FRAMES = 4800;      // 100ms buffer (minimal)
    constexpr int CONSOLE_INTERVAL_SEC = 60;          
    constexpr int POLL_INTERVAL_MS = 5;               
    constexpr bool ENABLE_VAD_DEFAULT = true;         // VAD essential
    constexpr bool LOW_POWER_DEFAULT = true;          
    constexpr int LOG_EVERY_N_FRAMES = 500;           // Log every 10 seconds
    constexpr float VAD_THRESHOLD = 0.001f;           
    constexpr int VAD_HANGOVER_FRAMES = 20;           // 200ms hangover
    
    #define ENABLE_WAV_WRITING 1
    #define ENABLE_FILE_LOGGING 1
    #define ENABLE_CONSOLE_OUTPUT 1
    
    #define PROFILE_NAME "EMBEDDED"

#else  // BUILD_DESKTOP (default)
    // ============================================================
    // DESKTOP PROFILE: Full features, quality first
    // ============================================================
    // Target: Studio recording, conference rooms, research
    // Focus: Maximum quality and flexibility
    
    constexpr int SAMPLE_RATE = 48000;                // 48kHz
    constexpr int FRAME_SIZE = 480;                   // 10ms frames
    constexpr int NUM_CHANNELS_MAX = 16;              // Many channels
    constexpr int CIRCULAR_BUFFER_FRAMES = 48000;     // 1000ms buffer (stable)
    constexpr int CONSOLE_INTERVAL_SEC = 10;          
    constexpr int POLL_INTERVAL_MS = 1;               
    constexpr bool ENABLE_VAD_DEFAULT = false;        // Always process
    constexpr bool LOW_POWER_DEFAULT = false;         
    constexpr int LOG_EVERY_N_FRAMES = 1;             // Log every frame
    constexpr float VAD_THRESHOLD = 0.001f;           
    constexpr int VAD_HANGOVER_FRAMES = 20;           // 200ms hangover
    
    #define ENABLE_WAV_WRITING 1
    #define ENABLE_FILE_LOGGING 1
    #define ENABLE_CONSOLE_OUTPUT 1
    
    #define PROFILE_NAME "DESKTOP"

#endif

// ------------------ Common Constants ------------------
constexpr int NUM_CHANNELS_DEFAULT = 1;
constexpr float DENORMAL_THRESHOLD = 1.0e-30f;
constexpr float DENORMAL_GUARD_INITIAL = 1.0e-20f;

// ------------------ Low Power Mode ------------------
// When LOW_POWER_DEFAULT is true (Wearable/Embedded):
// - WAV file recording is disabled by default (use --wav flag to enable for debugging)
// - Logging frequency is reduced (LOG_EVERY_N_FRAMES)
// - Live audio processing continues normally
// - Saves disk I/O and battery power

// ------------------ Compile-Time Checks ------------------
static_assert(FRAME_SIZE > 0, "FRAME_SIZE must be positive");
static_assert(SAMPLE_RATE == 48000, "RNNoise requires 48kHz sample rate");
static_assert(NUM_CHANNELS_MAX > 0, "NUM_CHANNELS_MAX must be positive");
static_assert(CIRCULAR_BUFFER_FRAMES >= FRAME_SIZE, "Buffer must be at least one frame");

// ------------------ Calculated Values ------------------
constexpr int BUFFER_LATENCY_MS = (CIRCULAR_BUFFER_FRAMES * 1000) / SAMPLE_RATE;
constexpr float FRAME_DURATION_MS = (static_cast<float>(FRAME_SIZE) * 1000.0f) / SAMPLE_RATE;

// ------------------ Profile Summary ------------------
/*
 * PROFILE COMPARISON:
 * 
 * Feature        | Desktop  | Wearable | Embedded
 * ---------------|----------|----------|----------
 * Sample Rate    | 48 kHz   | 48 kHz   | 48 kHz (ideally should be 16 kHz)
 * Channels       | 16       | 8        | 1
 * Buffer         | 1000ms   | 200ms    | 100ms
 * VAD Default    | OFF      | ON       | ON
 * Low Power      | OFF      | ON       | ON
 * WAV Default    | ON       | OFF      | OFF
 * 
 * NOTE: All profiles use 48kHz because RNNoise requires it.
 *       For embedded systems, 16kHz would be more appropriate for power/bandwidth,
 *       but would require a different noise suppression algorithm.
 */

#endif // DENOISE_CONFIG_H