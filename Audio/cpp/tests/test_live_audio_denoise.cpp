/**
 * @file test_live_audio_denoise.cpp
 * @brief Unit tests for live audio denoising application
 * @author Julia Wen (wendigilane@gmail.com)
 * @date 12-03-2025
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cmath>
#include "../inc/SPSCFloatBuffer.h"
#include "../inc/denormal_control.h"
#include "../inc/wav_writer.h"
#include "rnnoise.h"

// ============================================================================
// Google Test Fixture
// ============================================================================

class LiveAudioDenoiseTest : public ::testing::Test {
protected:
    std::filesystem::path testOutputDir;

    void SetUp() override {
        testOutputDir = "test_audio_output";
        if (!std::filesystem::exists(testOutputDir)) {
            std::filesystem::create_directories(testOutputDir);
        }
    }

    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(testOutputDir)) {
            std::filesystem::remove_all(testOutputDir);
        }
    }

    // Helper: Generate sine wave test signal
    std::vector<float> generateSineWave(float frequency, int sampleRate, 
                                        int numSamples, float amplitude = 1.0f) const {
        std::vector<float> buffer(numSamples);
        for (int i = 0; i < numSamples; ++i) {
            buffer[i] = amplitude * std::sin(2.0f * M_PI * frequency * i / sampleRate);
        }
        return buffer;
    }

    // Helper: Generate white noise
    std::vector<float> generateWhiteNoise(int numSamples, float amplitude = 0.1f) const {
        std::vector<float> buffer(numSamples);
        for (int i = 0; i < numSamples; ++i) {
            buffer[i] = amplitude * (2.0f * (rand() / static_cast<float>(RAND_MAX)) - 1.0f);
        }
        return buffer;
    }

    // Helper: Calculate RMS
    float calculateRMS(const std::vector<float>& buffer) const {
        float sum = 0.0f;
        for (float sample : buffer) {
            sum += sample * sample;
        }
        return std::sqrt(sum / buffer.size());
    }

    // Helper: Interleave stereo samples
    std::vector<float> interleave(const std::vector<float>& left, 
                                  const std::vector<float>& right) const {
        std::vector<float> interleaved(left.size() + right.size());
        for (size_t i = 0; i < left.size(); ++i) {
            interleaved[i * 2] = left[i];
            interleaved[i * 2 + 1] = right[i];
        }
        return interleaved;
    }

    // Helper: De-interleave stereo samples
    void deinterleave(const std::vector<float>& interleaved,
                     std::vector<float>& left, 
                     std::vector<float>& right) const {
        size_t numFrames = interleaved.size() / 2;
        left.resize(numFrames);
        right.resize(numFrames);
        for (size_t i = 0; i < numFrames; ++i) {
            left[i] = interleaved[i * 2];
            right[i] = interleaved[i * 2 + 1];
        }
    }
};

// ============================================================================
// SPSC Buffer Tests
// ============================================================================

TEST_F(LiveAudioDenoiseTest, SPSCBuffer_BasicOperations) {
    SPSCFloatBuffer buffer(1000);
    
    // Test push and pop
    EXPECT_TRUE(buffer.push(1.0f));
    EXPECT_TRUE(buffer.push(2.0f));
    EXPECT_TRUE(buffer.push(3.0f));
    
    float val;
    EXPECT_TRUE(buffer.pop(val));
    EXPECT_FLOAT_EQ(val, 1.0f);
    EXPECT_TRUE(buffer.pop(val));
    EXPECT_FLOAT_EQ(val, 2.0f);
    EXPECT_TRUE(buffer.pop(val));
    EXPECT_FLOAT_EQ(val, 3.0f);
}

TEST_F(LiveAudioDenoiseTest, SPSCBuffer_BulkOperations) {
    SPSCFloatBuffer buffer(1000);
    
    std::vector<float> testData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    // Test bulk push
    size_t pushed = buffer.pushBulk(testData.data(), testData.size());
    EXPECT_EQ(pushed, testData.size());
    EXPECT_EQ(buffer.available(), testData.size());
    
    // Test bulk pop
    std::vector<float> output(testData.size());
    size_t popped = buffer.popBulk(output.data(), output.size());
    EXPECT_EQ(popped, testData.size());
    
    for (size_t i = 0; i < testData.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], testData[i]);
    }
}

TEST_F(LiveAudioDenoiseTest, SPSCBuffer_Overflow) {
    SPSCFloatBuffer buffer(5);

    // Use the buffer's actual usable capacity
    size_t usable = buffer.capacity();

    // Fill buffer
    for (size_t i = 0; i < usable; ++i) {
        EXPECT_TRUE(buffer.push(static_cast<float>(i)));
    }

    // Try to push when full (should fail)
    EXPECT_FALSE(buffer.push(99.0f));
}

TEST_F(LiveAudioDenoiseTest, SPSCBuffer_Underflow) {
    SPSCFloatBuffer buffer(5);
    
    float val;
    // Try to pop from empty buffer (should fail)
    EXPECT_FALSE(buffer.pop(val));
}

TEST_F(LiveAudioDenoiseTest, SPSCBuffer_Wraparound) {
    SPSCFloatBuffer buffer(10);
    
    // Push and pop to advance pointers
    for (int i = 0; i < 8; ++i) {
        buffer.push(static_cast<float>(i));
    }
    
    for (int i = 0; i < 8; ++i) {
        float val;
        buffer.pop(val);
    }
    
    // Now push again to test wraparound
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(buffer.push(static_cast<float>(i * 10)));
    }
    
    EXPECT_EQ(buffer.available(), 10);
}

// ============================================================================
// Denormal Control Tests
// ============================================================================

TEST_F(LiveAudioDenoiseTest, DenormalControl_GuardFunction) {
    float denormalValue = 1e-40f;
    float normalValue = 0.5f;
    float guardValue = 1e-20f;
    
    // Guard should add to denormal values
    float guarded = denormal_control::guardDenormal(denormalValue, guardValue);
    EXPECT_NE(guarded, denormalValue);
    EXPECT_EQ(guarded, denormalValue + guardValue);
    
    // Guard should not affect normal values significantly
    float guardedNormal = denormal_control::guardDenormal(normalValue, guardValue);
    EXPECT_NEAR(guardedNormal, normalValue + guardValue, 1e-15f);
}

TEST_F(LiveAudioDenoiseTest, DenormalControl_ThresholdClamping) {
    constexpr float DENORMAL_THRESHOLD = 1.0e-30f;
    
    // Very small positive value
    float smallPos = 1.0e-35f;
    if (smallPos > -DENORMAL_THRESHOLD && smallPos < DENORMAL_THRESHOLD) {
        smallPos = 0.0f;
    }
    EXPECT_FLOAT_EQ(smallPos, 0.0f);
    
    // Very small negative value
    float smallNeg = -1.0e-35f;
    if (smallNeg > -DENORMAL_THRESHOLD && smallNeg < DENORMAL_THRESHOLD) {
        smallNeg = 0.0f;
    }
    EXPECT_FLOAT_EQ(smallNeg, 0.0f);
    
    // Normal values should pass through
    float normal = 0.1f;
    if (normal > -DENORMAL_THRESHOLD && normal < DENORMAL_THRESHOLD) {
        normal = 0.0f;
    }
    EXPECT_FLOAT_EQ(normal, 0.1f);
}

// ============================================================================
// RNNoise Integration Tests
// ============================================================================

TEST_F(LiveAudioDenoiseTest, RNNoise_CreateDestroy) {
    DenoiseState* state = rnnoise_create(nullptr);
    ASSERT_NE(state, nullptr);
    rnnoise_destroy(state);
}

TEST_F(LiveAudioDenoiseTest, RNNoise_MultipleStates) {
    constexpr int NUM_CHANNELS = 4;
    std::vector<DenoiseState*> states(NUM_CHANNELS);
    
    // Create multiple states
    for (int i = 0; i < NUM_CHANNELS; ++i) {
        states[i] = rnnoise_create(nullptr);
        ASSERT_NE(states[i], nullptr);
    }
    
    // Destroy all states
    for (auto* state : states) {
        rnnoise_destroy(state);
    }
}

TEST_F(LiveAudioDenoiseTest, RNNoise_ProcessFrame_Mono) {
    constexpr int FRAME_SIZE = 480;
    DenoiseState* state = rnnoise_create(nullptr);
    ASSERT_NE(state, nullptr);
    
    // Generate test input (sine + noise)
    auto sine = generateSineWave(440.0f, 48000, FRAME_SIZE, 0.5f);
    auto noise = generateWhiteNoise(FRAME_SIZE, 0.1f);
    
    std::vector<float> input(FRAME_SIZE);
    std::vector<float> output(FRAME_SIZE);
    
    for (int i = 0; i < FRAME_SIZE; ++i) {
        input[i] = sine[i] + noise[i];
    }
    
    // Process frame
    rnnoise_process_frame(state, output.data(), input.data());
    
    // Output should be different from input (denoising occurred)
    bool isDifferent = false;
    for (int i = 0; i < FRAME_SIZE; ++i) {
        if (std::abs(output[i] - input[i]) > 1e-6f) {
            isDifferent = true;
            break;
        }
    }
    EXPECT_TRUE(isDifferent);
    
    rnnoise_destroy(state);
}

TEST_F(LiveAudioDenoiseTest, RNNoise_ProcessFrame_Stereo) {
    constexpr int FRAME_SIZE = 480;
    std::vector<DenoiseState*> states = {
        rnnoise_create(nullptr),
        rnnoise_create(nullptr)
    };
    
    ASSERT_NE(states[0], nullptr);
    ASSERT_NE(states[1], nullptr);
    
    // Generate test data for left and right channels
    auto leftSine = generateSineWave(440.0f, 48000, FRAME_SIZE, 0.5f);
    auto rightSine = generateSineWave(880.0f, 48000, FRAME_SIZE, 0.5f);
    auto noise = generateWhiteNoise(FRAME_SIZE, 0.1f);
    
    std::vector<float> leftIn(FRAME_SIZE), rightIn(FRAME_SIZE);
    std::vector<float> leftOut(FRAME_SIZE), rightOut(FRAME_SIZE);
    
    for (int i = 0; i < FRAME_SIZE; ++i) {
        leftIn[i] = leftSine[i] + noise[i];
        rightIn[i] = rightSine[i] + noise[i];
    }
    
    // Process each channel independently
    rnnoise_process_frame(states[0], leftOut.data(), leftIn.data());
    rnnoise_process_frame(states[1], rightOut.data(), rightIn.data());
    
    // Both outputs should be different from inputs
    bool leftDifferent = false, rightDifferent = false;
    for (int i = 0; i < FRAME_SIZE; ++i) {
        if (std::abs(leftOut[i] - leftIn[i]) > 1e-6f) leftDifferent = true;
        if (std::abs(rightOut[i] - rightIn[i]) > 1e-6f) rightDifferent = true;
    }
    EXPECT_TRUE(leftDifferent);
    EXPECT_TRUE(rightDifferent);
    
    for (auto* state : states) {
        rnnoise_destroy(state);
    }
}

// ============================================================================
// Audio Processing Pipeline Tests
// ============================================================================

TEST_F(LiveAudioDenoiseTest, Pipeline_DeinterleaveProcess) {
    constexpr int FRAME_SIZE = 480;
    constexpr int NUM_CHANNELS = 2;
    
    // Generate stereo input
    auto leftSine = generateSineWave(440.0f, 48000, FRAME_SIZE);
    auto rightSine = generateSineWave(880.0f, 48000, FRAME_SIZE);
    auto interleaved = interleave(leftSine, rightSine);
    
    // De-interleave
    std::vector<float> leftCh(FRAME_SIZE), rightCh(FRAME_SIZE);
    for (int i = 0; i < FRAME_SIZE; ++i) {
        leftCh[i] = interleaved[i * NUM_CHANNELS + 0];
        rightCh[i] = interleaved[i * NUM_CHANNELS + 1];
    }
    
    // Verify de-interleaving
    for (int i = 0; i < FRAME_SIZE; ++i) {
        EXPECT_FLOAT_EQ(leftCh[i], leftSine[i]);
        EXPECT_FLOAT_EQ(rightCh[i], rightSine[i]);
    }
}

TEST_F(LiveAudioDenoiseTest, Pipeline_RMSCalculation) {
    constexpr int FRAME_SIZE = 480;
    constexpr int NUM_CHANNELS = 2;
    
    auto sine = generateSineWave(440.0f, 48000, FRAME_SIZE, 1.0f);
    auto interleaved = interleave(sine, sine);
    
    // Calculate RMS as in the application
    float rms = 0.0f;
    for (size_t i = 0; i < interleaved.size(); i += NUM_CHANNELS) {
        for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
            rms += interleaved[i + ch] * interleaved[i + ch];
        }
    }
    rms = std::sqrt(rms / interleaved.size());
    
    // For a full-scale sine wave, RMS should be approximately 1/sqrt(2)
    EXPECT_NEAR(rms, 1.0f / std::sqrt(2.0f), 0.01f);
}

TEST_F(LiveAudioDenoiseTest, Pipeline_BypassMode) {
    constexpr int FRAME_SIZE = 480;
    bool bypassDenoise = true;
    
    auto input = generateSineWave(440.0f, 48000, FRAME_SIZE);
    std::vector<float> processedOutput(FRAME_SIZE);
    std::vector<float> finalOutput(FRAME_SIZE);
    
    // Simulate RNNoise processing
    DenoiseState* state = rnnoise_create(nullptr);
    rnnoise_process_frame(state, processedOutput.data(), input.data());
    
    // Apply bypass logic
    for (int i = 0; i < FRAME_SIZE; ++i) {
        finalOutput[i] = bypassDenoise ? input[i] : processedOutput[i];
    }
    
    // In bypass mode, output should match input exactly
    for (int i = 0; i < FRAME_SIZE; ++i) {
        EXPECT_FLOAT_EQ(finalOutput[i], input[i]);
    }
    
    rnnoise_destroy(state);
}

// ============================================================================
// WAV Writer Integration Tests
// ============================================================================

TEST_F(LiveAudioDenoiseTest, WavWriter_MonoOutput) {
    auto outputPath = testOutputDir / "test_mono.wav";
    constexpr int SAMPLE_RATE = 48000;
    constexpr int NUM_CHANNELS = 1;
    constexpr int BIT_DEPTH = 16;
    
    WavWriter writer(outputPath.string(), SAMPLE_RATE, NUM_CHANNELS, BIT_DEPTH);
    
    // Write some test data
    auto testData = generateSineWave(440.0f, SAMPLE_RATE, 480);
    for (float sample : testData) {
        writer.writeFrame(&sample);
    }
    
    writer.close();
    
    // Verify file was created
    EXPECT_TRUE(std::filesystem::exists(outputPath));
    EXPECT_GT(std::filesystem::file_size(outputPath), 0);
}

TEST_F(LiveAudioDenoiseTest, WavWriter_StereoOutput) {
    auto outputPath = testOutputDir / "test_stereo.wav";
    constexpr int SAMPLE_RATE = 48000;
    constexpr int NUM_CHANNELS = 2;
    constexpr int BIT_DEPTH = 16;
    
    WavWriter writer(outputPath.string(), SAMPLE_RATE, NUM_CHANNELS, BIT_DEPTH);
    
    // Write stereo test data
    auto leftData = generateSineWave(440.0f, SAMPLE_RATE, 480);
    auto rightData = generateSineWave(880.0f, SAMPLE_RATE, 480);
    auto stereoData = interleave(leftData, rightData);
    
    for (size_t i = 0; i < stereoData.size(); i += NUM_CHANNELS) {
        writer.writeFrame(&stereoData[i]);
    }
    
    writer.close();
    
    // Verify file was created
    EXPECT_TRUE(std::filesystem::exists(outputPath));
    EXPECT_GT(std::filesystem::file_size(outputPath), 0);
}

TEST_F(LiveAudioDenoiseTest, WavWriter_MultipleBitDepths) {
    constexpr int SAMPLE_RATE = 48000;
    constexpr int NUM_CHANNELS = 1;
    auto testData = generateSineWave(440.0f, SAMPLE_RATE, 480);
    
    for (int bitDepth : {16, 24, 32}) {
        auto outputPath = testOutputDir / ("test_" + std::to_string(bitDepth) + "bit.wav");
        
        WavWriter writer(outputPath.string(), SAMPLE_RATE, NUM_CHANNELS, bitDepth);
        
        for (float sample : testData) {
            writer.writeFrame(&sample);
        }
        
        writer.close();
        
        EXPECT_TRUE(std::filesystem::exists(outputPath));
        EXPECT_GT(std::filesystem::file_size(outputPath), 0);
    }
}

// ============================================================================
// File Output Tests
// ============================================================================

TEST_F(LiveAudioDenoiseTest, FileOutput_LogFile) {
    auto logPath = testOutputDir / "test_rms_log.txt";
    std::ofstream logFile(logPath, std::ios::out);
    
    ASSERT_TRUE(logFile.is_open());
    
    // Write some test log data
    for (int i = 1; i <= 100; ++i) {
        float inRMS = 0.1f * i;
        float outRMS = 0.08f * i;
        logFile << i << " " << inRMS << " " << outRMS << "\n";
    }
    
    logFile.close();
    
    // Verify file was created and has content
    EXPECT_TRUE(std::filesystem::exists(logPath));
    
    // Read back and verify
    std::ifstream readLog(logPath);
    int frameNum;
    float inRMS, outRMS;
    int lineCount = 0;
    
    while (readLog >> frameNum >> inRMS >> outRMS) {
        lineCount++;
    }
    
    EXPECT_EQ(lineCount, 100);
}

TEST_F(LiveAudioDenoiseTest, FileOutput_DirectoryCreation) {
    auto nestedDir = testOutputDir / "nested" / "output";
    
    // Create nested directories
    std::filesystem::create_directories(nestedDir);
    
    EXPECT_TRUE(std::filesystem::exists(nestedDir));
    EXPECT_TRUE(std::filesystem::is_directory(nestedDir));
}

// ============================================================================
// End-to-End Simulation Tests
// ============================================================================

TEST_F(LiveAudioDenoiseTest, EndToEnd_MonoProcessing) {
    constexpr int FRAME_SIZE = 480;
    constexpr int SAMPLE_RATE = 48000;
    constexpr int NUM_CHANNELS = 1;
    constexpr int NUM_FRAMES = 10;
    
    // Setup
    SPSCFloatBuffer inputBuffer(FRAME_SIZE * NUM_CHANNELS * 100);
    SPSCFloatBuffer outputBuffer(FRAME_SIZE * NUM_CHANNELS * 100);
    DenoiseState* state = rnnoise_create(nullptr);
    
    // Generate and push test data
    for (int f = 0; f < NUM_FRAMES; ++f) {
        auto frameData = generateSineWave(440.0f, SAMPLE_RATE, FRAME_SIZE, 0.5f);
        inputBuffer.pushBulk(frameData.data(), frameData.size());
    }
    
    // Process frames
    std::vector<float> inFrame(FRAME_SIZE);
    std::vector<float> outFrame(FRAME_SIZE);
    
    for (int f = 0; f < NUM_FRAMES; ++f) {
        inputBuffer.popBulk(inFrame.data(), FRAME_SIZE);
        rnnoise_process_frame(state, outFrame.data(), inFrame.data());
        outputBuffer.pushBulk(outFrame.data(), FRAME_SIZE);
    }
    
    // Verify data was processed
    EXPECT_EQ(outputBuffer.available(), FRAME_SIZE * NUM_FRAMES);
    
    rnnoise_destroy(state);
}

TEST_F(LiveAudioDenoiseTest, EndToEnd_StereoProcessing) {
    constexpr int FRAME_SIZE = 480;
    constexpr int SAMPLE_RATE = 48000;
    constexpr int NUM_CHANNELS = 2;
    constexpr int NUM_FRAMES = 10;
    
    // Setup
    SPSCFloatBuffer inputBuffer(FRAME_SIZE * NUM_CHANNELS * 100);
    SPSCFloatBuffer outputBuffer(FRAME_SIZE * NUM_CHANNELS * 100);
    
    std::vector<DenoiseState*> states = {
        rnnoise_create(nullptr),
        rnnoise_create(nullptr)
    };
    
    // Generate and push stereo test data
    for (int f = 0; f < NUM_FRAMES; ++f) {
        auto leftData = generateSineWave(440.0f, SAMPLE_RATE, FRAME_SIZE);
        auto rightData = generateSineWave(880.0f, SAMPLE_RATE, FRAME_SIZE);
        auto stereoData = interleave(leftData, rightData);
        inputBuffer.pushBulk(stereoData.data(), stereoData.size());
    }
    
    // Process frames
    std::vector<float> inFrame(FRAME_SIZE * NUM_CHANNELS);
    std::vector<float> outFrame(FRAME_SIZE * NUM_CHANNELS);
    std::vector<float> inCh(FRAME_SIZE);
    std::vector<float> outCh(FRAME_SIZE);
    
    for (int f = 0; f < NUM_FRAMES; ++f) {
        inputBuffer.popBulk(inFrame.data(), FRAME_SIZE * NUM_CHANNELS);
        
        // Process each channel
        for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
            // De-interleave
            for (int i = 0; i < FRAME_SIZE; ++i) {
                inCh[i] = inFrame[i * NUM_CHANNELS + ch];
            }
            
            // Process
            rnnoise_process_frame(states[ch], outCh.data(), inCh.data());
            
            // Re-interleave
            for (int i = 0; i < FRAME_SIZE; ++i) {
                outFrame[i * NUM_CHANNELS + ch] = outCh[i];
            }
        }
        
        outputBuffer.pushBulk(outFrame.data(), FRAME_SIZE * NUM_CHANNELS);
    }
    
    // Verify data was processed
    EXPECT_EQ(outputBuffer.available(), FRAME_SIZE * NUM_CHANNELS * NUM_FRAMES);
    
    for (auto* state : states) {
        rnnoise_destroy(state);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}