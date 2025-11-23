/**
 * @file fft_utils.cpp
 * @brief Fast Fourier Transform utilities (implementation)
 * @author Julia Wen (wendigilane@gmail.com)
 * @date 2025-11-21
 */

#include "../inc/fft_utils.h"
#include <cmath>
#include <algorithm>
#include "../inc/error_codes.h"

namespace fft {

const double kPI = 3.14159265358979323846;

// ============================================================================
// Bit Reversal Permutation
// ============================================================================

void bitReverseInPlace(std::vector<cd>& a) noexcept {
    const size_t n = a.size();
    if (n <= 2) return;
    
    size_t j = 0;
    for (size_t i = 1; i < n - 1; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

// ============================================================================
// FFT Implementation
// ============================================================================

int compute(std::vector<cd>& a, bool invert) noexcept {
    const size_t n = a.size();
    
    // Validate input
    if (n == 0) return ERR_FFT_COMPUTE;
    if (n == 1) return SUCCESS;
    if ((n & (n - 1)) != 0) return ERR_FFT_COMPUTE;  // not a power of two
    
    // Bit-reversal permutation
    bitReverseInPlace(a);
    
    // FFT computation
    const double sgn = invert ? -1.0 : 1.0;
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = sgn * 2.0 * kPI / static_cast<double>(len);
        const cd wlen(std::cos(ang), std::sin(ang));
        
        // Process each block of size 'len'
        for (size_t i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            const size_t half = len >> 1;
            cd* a0 = &a[i];          // local pointer avoids repeated indexing
            cd* a1 = a0 + half;
            
            for (size_t j = 0; j < half; ++j) {
                const cd u = a0[j];
                const cd v = a1[j] * w;
                a0[j] = u + v;
                a1[j] = u - v;
                w *= wlen;
            }
        }
    }
    
    // Normalize for inverse FFT
    if (invert) {
        const double invN = 1.0 / static_cast<double>(n);
        for (auto& x : a) x *= invN;
    }
    
    return SUCCESS;
}

// ============================================================================
// Helper Functions
// ============================================================================

size_t nextPowerOf2(size_t n) {
    size_t power = 1;
    while (power < n) power <<= 1;
    return power;
}

std::vector<cd> prepareForFFT(const std::vector<double>& samples) {
    size_t fftSize = nextPowerOf2(samples.size());
    std::vector<cd> data(fftSize, 0.0);
    std::copy(samples.begin(), samples.end(), data.begin());
    return data;
}

std::vector<double> getMagnitudeSpectrum(const std::vector<cd>& fftData) {
    const size_t halfSize = fftData.size() / 2;
    std::vector<double> magnitudes(halfSize);
    for (size_t i = 0; i < halfSize; ++i) {
        magnitudes[i] = std::abs(fftData[i]);
    }
    return magnitudes;
}

std::vector<double> getFrequencyBins(size_t fftSize, uint32_t sampleRate) {
    const size_t halfSize = fftSize / 2;
    std::vector<double> frequencies(halfSize);
    for (size_t i = 0; i < halfSize; ++i) {
        frequencies[i] = static_cast<double>(i) * static_cast<double>(sampleRate) / static_cast<double>(fftSize);
    }
    return frequencies;
}

} // namespace fft
