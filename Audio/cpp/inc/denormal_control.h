#pragma once
/**
 * @file denormal_control.h
 * @brief Cross-platform denormal floating-point control
 * Supports x86 SSE, ARM64 (Apple/Linux), and software fallback
 * @author Julia Wen (wendigilane@gmail.com)
 * @date 2025-11-25
 */
#include <cstdint>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <xmmintrin.h>
    #define DC_HAS_X86_SSE 1
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    #include <fenv.h>
    #define DC_HAS_ARM_FP 1
#endif

namespace denormal_control {

// -----------------------------------------------------------------------------
// ARM FPCR bit definitions
// -----------------------------------------------------------------------------
constexpr uint64_t ARM_FPCR_FZ = (1ULL << 24);

// -----------------------------------------------------------------------------
// disableDenormals() — enable hardware flush-to-zero / denormals-are-zero
// -----------------------------------------------------------------------------
inline void disableDenormals() {

#if defined(DC_HAS_X86_SSE)
    // x86 SSE: FTZ + DAZ
    unsigned int mxcsr = _mm_getcsr();
    mxcsr |= (1 << 15);  // FTZ
    mxcsr |= (1 << 6);   // DAZ
    _mm_setcsr(mxcsr);

#elif defined(DC_HAS_ARM_FP)
    // macOS / Linux ARM64
    #if defined(__APPLE__) && defined(__aarch64__)
        uint64_t fpcr;
        asm volatile("mrs %0, fpcr" : "=r"(fpcr));
        fpcr |= ARM_FPCR_FZ;
        asm volatile("msr fpcr, %0" : : "r"(fpcr));
    #elif defined(__linux__) && defined(__aarch64__)
        uint64_t fpcr;
        asm volatile("mrs %0, fpcr" : "=r"(fpcr));
        fpcr |= ARM_FPCR_FZ;
        asm volatile("msr fpcr, %0" : : "r"(fpcr));
    #else
        // Fallback ARM
        fenv_t env;
        fegetenv(&env);
        env.__fpcr |= ARM_FPCR_FZ;
        fesetenv(&env);
    #endif

#else
    // No hardware support — nothing to do
#endif
}

// -----------------------------------------------------------------------------
// software guard to prevent denormals
// -----------------------------------------------------------------------------
inline float guardDenormal(float value, float guard = 1.0e-30f) {
    value += guard;
    if (std::fabs(value) < guard) value = 0.0f;
    return value;
}

// -----------------------------------------------------------------------------
// RAII class: automatically disables denormals in a scope
// -----------------------------------------------------------------------------
struct AutoDisable {
    AutoDisable() { disableDenormals(); }
};

} // namespace denormal_control

