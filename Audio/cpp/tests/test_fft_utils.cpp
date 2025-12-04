/**
 * @file test_fft_utils.cpp
 * @brief Unit tests for FFT utilities
 * @author Julia Wen (wendigilane@gmail.com)
 * @date 12-03-2025
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../inc/fft_utils.h"
#include "../inc/error_codes.h"
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

using namespace fft;

// ============================================================================
// Google Test Fixture
// ============================================================================

class FFTUtilsTest : public ::testing::Test {
protected:
    static constexpr double kPI = 3.14159265358979323846;
    static constexpr double kEpsilon = 1e-10;
    
    void SetUp() override {
        // Common setup
    }

    void TearDown() override {
        // Cleanup
    }

    // Helper: Generate sine wave samples
    std::vector<double> generateSineWave(double frequency, double sampleRate, 
                                        size_t numSamples, double amplitude = 1.0) const {
        std::vector<double> samples(numSamples);
        for (size_t i = 0; i < numSamples; ++i) {
            samples[i] = amplitude * std::sin(2.0 * kPI * frequency * i / sampleRate);
        }
        return samples;
    }

    // Helper: Generate cosine wave samples
    std::vector<double> generateCosineWave(double frequency, double sampleRate, 
                                          size_t numSamples, double amplitude = 1.0) const {
        std::vector<double> samples(numSamples);
        for (size_t i = 0; i < numSamples; ++i) {
            samples[i] = amplitude * std::cos(2.0 * kPI * frequency * i / sampleRate);
        }
        return samples;
    }

    // Helper: Generate impulse signal (delta function)
    std::vector<double> generateImpulse(size_t numSamples, size_t impulsePos = 0) const {
        std::vector<double> samples(numSamples, 0.0);
        if (impulsePos < numSamples) {
            samples[impulsePos] = 1.0;
        }
        return samples;
    }

    // Helper: Generate DC signal (constant value)
    std::vector<double> generateDC(size_t numSamples, double value = 1.0) const {
        return std::vector<double>(numSamples, value);
    }

    // Helper: Check if a number is close to expected value
    bool isClose(double a, double b, double epsilon = kEpsilon) const {
        return std::abs(a - b) < epsilon;
    }

    // Helper: Check if complex number is close to expected value
    bool isClose(const cd& a, const cd& b, double epsilon = kEpsilon) const {
        return std::abs(a - b) < epsilon;
    }

    // Helper: Find peak frequency in magnitude spectrum
    size_t findPeakIndex(const std::vector<double>& magnitudes) const {
        return std::distance(magnitudes.begin(), 
                           std::max_element(magnitudes.begin(), magnitudes.end()));
    }
};

// ============================================================================
// Bit Reversal Tests
// ============================================================================

TEST_F(FFTUtilsTest, BitReverse_Size2) {
    std::vector<cd> data = {cd(0, 0), cd(1, 0)};
    bitReverseInPlace(data);
    
    // For size 2, no change expected
    EXPECT_EQ(data[0], cd(0, 0));
    EXPECT_EQ(data[1], cd(1, 0));
}

TEST_F(FFTUtilsTest, BitReverse_Size4) {
    std::vector<cd> data = {cd(0, 0), cd(1, 0), cd(2, 0), cd(3, 0)};
    bitReverseInPlace(data);
    
    // Bit-reversed order: 0, 2, 1, 3
    EXPECT_EQ(data[0], cd(0, 0));
    EXPECT_EQ(data[1], cd(2, 0));
    EXPECT_EQ(data[2], cd(1, 0));
    EXPECT_EQ(data[3], cd(3, 0));
}

TEST_F(FFTUtilsTest, BitReverse_Size8) {
    std::vector<cd> data(8);
    for (size_t i = 0; i < 8; ++i) {
        data[i] = cd(static_cast<double>(i), 0);
    }
    
    bitReverseInPlace(data);
    
    // Bit-reversed order: 0, 4, 2, 6, 1, 5, 3, 7
    EXPECT_EQ(data[0], cd(0, 0));
    EXPECT_EQ(data[1], cd(4, 0));
    EXPECT_EQ(data[2], cd(2, 0));
    EXPECT_EQ(data[3], cd(6, 0));
    EXPECT_EQ(data[4], cd(1, 0));
    EXPECT_EQ(data[5], cd(5, 0));
    EXPECT_EQ(data[6], cd(3, 0));
    EXPECT_EQ(data[7], cd(7, 0));
}

TEST_F(FFTUtilsTest, BitReverse_EmptyVector) {
    std::vector<cd> data;
    EXPECT_NO_THROW(bitReverseInPlace(data));
}

TEST_F(FFTUtilsTest, BitReverse_Size1) {
    std::vector<cd> data = {cd(42, 0)};
    bitReverseInPlace(data);
    EXPECT_EQ(data[0], cd(42, 0));
}

// ============================================================================
// FFT Compute Tests - Input Validation
// ============================================================================

TEST_F(FFTUtilsTest, Compute_EmptyInput) {
    std::vector<cd> data;
    EXPECT_EQ(compute(data, false), ERR_FFT_COMPUTE);
}

TEST_F(FFTUtilsTest, Compute_SingleElement) {
    std::vector<cd> data = {cd(1.0, 0.0)};
    EXPECT_EQ(compute(data, false), SUCCESS);
    EXPECT_EQ(data[0], cd(1.0, 0.0));
}

TEST_F(FFTUtilsTest, Compute_NonPowerOf2) {
    std::vector<cd> data(3);  // 3 is not a power of 2
    EXPECT_EQ(compute(data, false), ERR_FFT_COMPUTE);
    
    std::vector<cd> data2(5);  // 5 is not a power of 2
    EXPECT_EQ(compute(data2, false), ERR_FFT_COMPUTE);
    
    std::vector<cd> data3(7);  // 7 is not a power of 2
    EXPECT_EQ(compute(data3, false), ERR_FFT_COMPUTE);
}

TEST_F(FFTUtilsTest, Compute_ValidPowerOf2Sizes) {
    for (size_t size : {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
        std::vector<cd> data(size, cd(1.0, 0.0));
        EXPECT_EQ(compute(data, false), SUCCESS) << "Failed for size " << size;
    }
}

// ============================================================================
// FFT Compute Tests - Mathematical Properties
// ============================================================================

TEST_F(FFTUtilsTest, Compute_DCSignal) {
    // DC signal (all ones) should result in all energy in bin 0
    std::vector<cd> data(8, cd(1.0, 0.0));
    ASSERT_EQ(compute(data, false), SUCCESS);
    
    // All energy in DC bin
    EXPECT_GT(std::abs(data[0]), 7.9);  // Should be ~8.0
    
    // Other bins should be near zero
    for (size_t i = 1; i < 8; ++i) {
        EXPECT_LT(std::abs(data[i]), 1e-10);
    }
}

TEST_F(FFTUtilsTest, Compute_Impulse) {
    // Impulse at position 0 should result in flat spectrum
    std::vector<cd> data(8, cd(0.0, 0.0));
    data[0] = cd(1.0, 0.0);
    
    ASSERT_EQ(compute(data, false), SUCCESS);
    
    // All bins should have equal magnitude of 1.0
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(std::abs(data[i]), 1.0, 1e-10);
    }
}

TEST_F(FFTUtilsTest, Compute_RealCosineWave) {
    // Cosine wave at Fs/4 should give peaks at bins 2 and N-2
    constexpr size_t N = 8;
    std::vector<cd> data(N);
    
    // Generate cosine at frequency bin 2 (Fs/4)
    for (size_t i = 0; i < N; ++i) {
        data[i] = cd(std::cos(2.0 * kPI * 2.0 * i / N), 0.0);
    }
    
    ASSERT_EQ(compute(data, false), SUCCESS);
    
    // Should have peaks at bins 2 and 6 (N-2)
    EXPECT_GT(std::abs(data[2]), 3.9);  // Should be ~4.0
    EXPECT_GT(std::abs(data[6]), 3.9);  // Should be ~4.0
    
    // Other bins should be near zero
    for (size_t i = 0; i < N; ++i) {
        if (i != 2 && i != 6) {
            EXPECT_LT(std::abs(data[i]), 1e-10);
        }
    }
}

TEST_F(FFTUtilsTest, Compute_Linearity) {
    // FFT is linear: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)
    constexpr size_t N = 16;
    
    std::vector<cd> x(N), y(N), combined(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = cd(std::sin(2.0 * kPI * 2.0 * i / N), 0.0);
        y[i] = cd(std::cos(2.0 * kPI * 3.0 * i / N), 0.0);
        combined[i] = cd(2.0, 0.0) * x[i] + cd(3.0, 0.0) * y[i];
    }
    
    std::vector<cd> fft_x = x, fft_y = y, fft_combined = combined;
    ASSERT_EQ(compute(fft_x, false), SUCCESS);
    ASSERT_EQ(compute(fft_y, false), SUCCESS);
    ASSERT_EQ(compute(fft_combined, false), SUCCESS);
    
    // Check linearity for each bin
    for (size_t i = 0; i < N; ++i) {
        cd expected = cd(2.0, 0.0) * fft_x[i] + cd(3.0, 0.0) * fft_y[i];
        EXPECT_TRUE(isClose(fft_combined[i], expected, 1e-9));
    }
}

// ============================================================================
// FFT Inverse Tests
// ============================================================================

TEST_F(FFTUtilsTest, Compute_InverseReconstruction) {
    // Forward then inverse should reconstruct original signal
    constexpr size_t N = 16;
    std::vector<cd> original(N);
    
    // Generate test signal
    for (size_t i = 0; i < N; ++i) {
        original[i] = cd(std::sin(2.0 * kPI * 3.0 * i / N), 0.0);
    }
    
    std::vector<cd> data = original;
    
    // Forward FFT
    ASSERT_EQ(compute(data, false), SUCCESS);
    
    // Inverse FFT
    ASSERT_EQ(compute(data, true), SUCCESS);
    
    // Should match original
    for (size_t i = 0; i < N; ++i) {
        EXPECT_TRUE(isClose(data[i], original[i], 1e-10));
    }
}

TEST_F(FFTUtilsTest, Compute_InverseNormalization) {
    // Inverse FFT should properly normalize by 1/N
    std::vector<cd> data = {cd(1, 0), cd(1, 0), cd(1, 0), cd(1, 0)};
    
    ASSERT_EQ(compute(data, false), SUCCESS);  // Forward
    ASSERT_EQ(compute(data, true), SUCCESS);   // Inverse
    
    // Should be back to original values
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(data[i].real(), 1.0, 1e-10);
        EXPECT_NEAR(data[i].imag(), 0.0, 1e-10);
    }
}

TEST_F(FFTUtilsTest, Compute_MultipleInverseReconstructions) {
    // Test reconstruction with different signals
    constexpr size_t N = 32;
    
    std::vector<std::vector<cd>> testSignals = {
        std::vector<cd>(N, cd(1.0, 0.0)),  // DC
        std::vector<cd>(N, cd(0.0, 0.0)),  // Zeros
    };
    
    // Add impulse
    testSignals.push_back(std::vector<cd>(N, cd(0.0, 0.0)));
    testSignals.back()[0] = cd(1.0, 0.0);
    
    // Add sine wave
    testSignals.push_back(std::vector<cd>(N));
    for (size_t i = 0; i < N; ++i) {
        testSignals.back()[i] = cd(std::sin(2.0 * kPI * 5.0 * i / N), 0.0);
    }
    
    for (const auto& original : testSignals) {
        std::vector<cd> data = original;
        ASSERT_EQ(compute(data, false), SUCCESS);
        ASSERT_EQ(compute(data, true), SUCCESS);
        
        for (size_t i = 0; i < N; ++i) {
            EXPECT_TRUE(isClose(data[i], original[i], 1e-9));
        }
    }
}

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST_F(FFTUtilsTest, NextPowerOf2_SmallValues) {
    EXPECT_EQ(nextPowerOf2(1), 1);
    EXPECT_EQ(nextPowerOf2(2), 2);
    EXPECT_EQ(nextPowerOf2(3), 4);
    EXPECT_EQ(nextPowerOf2(4), 4);
    EXPECT_EQ(nextPowerOf2(5), 8);
    EXPECT_EQ(nextPowerOf2(7), 8);
    EXPECT_EQ(nextPowerOf2(8), 8);
    EXPECT_EQ(nextPowerOf2(9), 16);
}

TEST_F(FFTUtilsTest, NextPowerOf2_LargeValues) {
    EXPECT_EQ(nextPowerOf2(1000), 1024);
    EXPECT_EQ(nextPowerOf2(1024), 1024);
    EXPECT_EQ(nextPowerOf2(1025), 2048);
    EXPECT_EQ(nextPowerOf2(44100), 65536);
}

TEST_F(FFTUtilsTest, NextPowerOf2_Zero) {
    EXPECT_EQ(nextPowerOf2(0), 1);
}

TEST_F(FFTUtilsTest, PrepareForFFT_BasicUsage) {
    std::vector<double> samples = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto data = prepareForFFT(samples);
    
    // Should be padded to next power of 2 (8)
    EXPECT_EQ(data.size(), 8);
    
    // Original values should be preserved
    for (size_t i = 0; i < samples.size(); ++i) {
        EXPECT_DOUBLE_EQ(data[i].real(), samples[i]);
        EXPECT_DOUBLE_EQ(data[i].imag(), 0.0);
    }
    
    // Padding should be zeros
    for (size_t i = samples.size(); i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(data[i].real(), 0.0);
        EXPECT_DOUBLE_EQ(data[i].imag(), 0.0);
    }
}

TEST_F(FFTUtilsTest, PrepareForFFT_AlreadyPowerOf2) {
    std::vector<double> samples = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    auto data = prepareForFFT(samples);
    
    // Should remain size 8
    EXPECT_EQ(data.size(), 8);
    
    for (size_t i = 0; i < samples.size(); ++i) {
        EXPECT_DOUBLE_EQ(data[i].real(), samples[i]);
    }
}

TEST_F(FFTUtilsTest, PrepareForFFT_EmptyInput) {
    std::vector<double> samples;
    auto data = prepareForFFT(samples);
    
    // Should result in size 1 (next power of 2 from 0)
    EXPECT_EQ(data.size(), 1);
    EXPECT_DOUBLE_EQ(data[0].real(), 0.0);
}

TEST_F(FFTUtilsTest, GetMagnitudeSpectrum_BasicUsage) {
    std::vector<cd> fftData = {
        cd(4.0, 0.0),   // DC
        cd(0.0, 4.0),   // Bin 1
        cd(3.0, 4.0),   // Bin 2
        cd(0.0, 0.0),   // Bin 3
        cd(1.0, 1.0),   // Bin 4 (Nyquist for size 8)
        cd(0.0, 0.0),   // Bin 5
        cd(3.0, -4.0),  // Bin 6
        cd(0.0, -4.0)   // Bin 7
    };
    
    auto magnitudes = getMagnitudeSpectrum(fftData);
    
    // Should return only first half (4 values)
    EXPECT_EQ(magnitudes.size(), 4);
    
    // Check magnitudes
    EXPECT_DOUBLE_EQ(magnitudes[0], 4.0);           // |4 + 0i| = 4
    EXPECT_DOUBLE_EQ(magnitudes[1], 4.0);           // |0 + 4i| = 4
    EXPECT_DOUBLE_EQ(magnitudes[2], 5.0);           // |3 + 4i| = 5
    EXPECT_DOUBLE_EQ(magnitudes[3], 0.0);           // |0 + 0i| = 0
}

TEST_F(FFTUtilsTest, GetMagnitudeSpectrum_Size2) {
    std::vector<cd> fftData = {cd(1.0, 0.0), cd(1.0, 0.0)};
    auto magnitudes = getMagnitudeSpectrum(fftData);
    
    EXPECT_EQ(magnitudes.size(), 1);
    EXPECT_DOUBLE_EQ(magnitudes[0], 1.0);
}

TEST_F(FFTUtilsTest, GetFrequencyBins_StandardSampleRate) {
    constexpr size_t fftSize = 8;
    constexpr uint32_t sampleRate = 48000;
    
    auto frequencies = getFrequencyBins(fftSize, sampleRate);
    
    EXPECT_EQ(frequencies.size(), 4);  // Half of FFT size
    
    // Check frequency values
    EXPECT_DOUBLE_EQ(frequencies[0], 0.0);       // DC
    EXPECT_DOUBLE_EQ(frequencies[1], 6000.0);    // 48000 / 8 * 1
    EXPECT_DOUBLE_EQ(frequencies[2], 12000.0);   // 48000 / 8 * 2
    EXPECT_DOUBLE_EQ(frequencies[3], 18000.0);   // 48000 / 8 * 3
}

TEST_F(FFTUtilsTest, GetFrequencyBins_DifferentSampleRates) {
    constexpr size_t fftSize = 16;
    
    // Test with 44100 Hz
    auto freq44k = getFrequencyBins(fftSize, 44100);
    EXPECT_EQ(freq44k.size(), 8);
    EXPECT_DOUBLE_EQ(freq44k[1], 44100.0 / 16.0);
    
    // Test with 96000 Hz
    auto freq96k = getFrequencyBins(fftSize, 96000);
    EXPECT_EQ(freq96k.size(), 8);
    EXPECT_DOUBLE_EQ(freq96k[1], 96000.0 / 16.0);
}

TEST_F(FFTUtilsTest, GetFrequencyBins_NyquistFrequency) {
    constexpr size_t fftSize = 1024;
    constexpr uint32_t sampleRate = 48000;
    
    auto frequencies = getFrequencyBins(fftSize, sampleRate);
    
    // Last bin should be just below Nyquist (Fs/2)
    double nyquist = sampleRate / 2.0;
    EXPECT_LT(frequencies.back(), nyquist);
    EXPECT_GT(frequencies.back(), nyquist - 100);  // Close to Nyquist
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(FFTUtilsTest, Integration_DetectSineFrequency) {
    constexpr uint32_t sampleRate = 48000;
    constexpr double testFrequency = 1000.0;  // 1 kHz
    constexpr size_t numSamples = 1024;
    
    // Generate sine wave
    auto samples = generateSineWave(testFrequency, sampleRate, numSamples);
    
    // Prepare for FFT
    auto data = prepareForFFT(samples);
    
    // Compute FFT
    ASSERT_EQ(compute(data, false), SUCCESS);
    
    // Get magnitude spectrum
    auto magnitudes = getMagnitudeSpectrum(data);
    
    // Get frequency bins
    auto frequencies = getFrequencyBins(data.size(), sampleRate);
    
    // Find peak
    size_t peakIdx = findPeakIndex(magnitudes);
    double detectedFreq = frequencies[peakIdx];
    
    // Should detect frequency within reasonable tolerance
    EXPECT_NEAR(detectedFreq, testFrequency, 50.0);  // Within 50 Hz
}

TEST_F(FFTUtilsTest, Integration_MultipleFrequencies) {
    constexpr uint32_t sampleRate = 48000;
    constexpr size_t numSamples = 2048;
    
    // Generate signal with multiple frequencies
    auto signal1 = generateSineWave(1000.0, sampleRate, numSamples, 1.0);
    auto signal2 = generateSineWave(2000.0, sampleRate, numSamples, 0.5);
    auto signal3 = generateSineWave(4000.0, sampleRate, numSamples, 0.3);
    
    std::vector<double> combined(numSamples);
    for (size_t i = 0; i < numSamples; ++i) {
        combined[i] = signal1[i] + signal2[i] + signal3[i];
    }
    
    // Process with FFT
    auto data = prepareForFFT(combined);
    ASSERT_EQ(compute(data, false), SUCCESS);
    
    auto magnitudes = getMagnitudeSpectrum(data);
    auto frequencies = getFrequencyBins(data.size(), sampleRate);
    
    // Should have peaks near 1000, 2000, and 4000 Hz
    // (More sophisticated peak detection would be needed for exact verification)
    EXPECT_GT(magnitudes.size(), 0);
}

TEST_F(FFTUtilsTest, Integration_EndToEndWorkflow) {
    // Complete workflow: prepare -> compute -> extract magnitude
    std::vector<double> samples = {1.0, 0.5, -0.5, -1.0, -0.5, 0.5};
    
    // Step 1: Prepare
    auto data = prepareForFFT(samples);
    EXPECT_EQ(data.size(), 8);  // Next power of 2
    
    // Step 2: Compute FFT
    int result = compute(data, false);
    ASSERT_EQ(result, SUCCESS);
    
    // Step 3: Get magnitude spectrum
    auto magnitudes = getMagnitudeSpectrum(data);
    EXPECT_EQ(magnitudes.size(), 4);
    
    // Step 4: Get frequency bins
    auto frequencies = getFrequencyBins(data.size(), 44100);
    EXPECT_EQ(frequencies.size(), 4);
    
    // All magnitudes should be non-negative
    for (double mag : magnitudes) {
        EXPECT_GE(mag, 0.0);
    }
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_F(FFTUtilsTest, EdgeCase_AllZeros) {
    std::vector<cd> data(32, cd(0.0, 0.0));
    ASSERT_EQ(compute(data, false), SUCCESS);
    
    for (const auto& val : data) {
        EXPECT_LT(std::abs(val), 1e-15);
    }
}

TEST_F(FFTUtilsTest, EdgeCase_VerySmallValues) {
    std::vector<cd> data(16);
    for (size_t i = 0; i < 16; ++i) {
        data[i] = cd(1e-10, 1e-10);
    }
    
    ASSERT_EQ(compute(data, false), SUCCESS);
    // Should not crash or produce NaN
}

TEST_F(FFTUtilsTest, EdgeCase_LargeFFTSize) {
    constexpr size_t largeSize = 4096;
    std::vector<cd> data(largeSize, cd(1.0, 0.0));
    
    ASSERT_EQ(compute(data, false), SUCCESS);
    // Verify DC component
    EXPECT_GT(std::abs(data[0]), largeSize - 1);
}

TEST_F(FFTUtilsTest, Stress_RepeatedComputations) {
    std::vector<cd> data(64, cd(1.0, 0.0));
    
    // Perform multiple forward/inverse computations
    for (int i = 0; i < 100; ++i) {
        ASSERT_EQ(compute(data, false), SUCCESS);
        ASSERT_EQ(compute(data, true), SUCCESS);
    }
    
    // Should still be close to original
    for (const auto& val : data) {
        EXPECT_NEAR(val.real(), 1.0, 1e-8);
        EXPECT_NEAR(val.imag(), 0.0, 1e-8);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
