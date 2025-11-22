/**
 * @file fft_utils.h
 * @brief Fast Fourier Transform utilities (header declarations)
 * @author Julia Wen (wendigilane@gmail.com)
 * @date 2025-11-21
 */

#ifndef FFT_UTILS_H
#define FFT_UTILS_H

#include <vector>
#include <complex>
#include <cstdint>

namespace fft {

using cd = std::complex<double>;

extern const double kPI;

// ============================================================================
// Bit Reversal Permutation
// ============================================================================

void bitReverseInPlace(std::vector<cd>& a) noexcept;

// ============================================================================
// FFT Implementation
// ============================================================================

/**
 * @brief Compute FFT using iterative radix-2 Cooley-Tukey algorithm
 * @param a Input/output array (size must be power of 2)
 * @param invert If true, compute inverse FFT
 * @return SUCCESS on success, error code on failure
 */
int compute(std::vector<cd>& a, bool invert = false) noexcept;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Get next power of 2 >= n
 */
size_t nextPowerOf2(size_t n);

/**
 * @brief Prepare data for FFT by padding to power of 2
 * @param samples Input samples
 * @return Complex array padded to power of 2
 */
std::vector<cd> prepareForFFT(const std::vector<double>& samples);

/**
 * @brief Compute magnitude spectrum
 * @param fftData FFT output data
 * @return Vector of magnitudes (first half only, due to symmetry)
 */
std::vector<double> getMagnitudeSpectrum(const std::vector<cd>& fftData);

/**
 * @brief Get frequency bins for spectrum
 * @param fftSize Size of FFT
 * @param sampleRate Sample rate in Hz
 * @return Vector of frequency values in Hz
 */
std::vector<double> getFrequencyBins(size_t fftSize, uint32_t sampleRate);

} // namespace fft

#endif // FFT_UTILS_H
