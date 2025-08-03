#pragma once
#include <vector>
#include "flatbuffers/flatbuffers.h"

namespace iv {

#if defined(__x86_64__)
/**
 * AVX-512 implementation of implied volatility calculation using Newton-Raphson method
 * with Brent's method as fallback.
 */
void compute_iv_avx512(
    const float& S,
    const flatbuffers::Vector<float>* K,
    const float& T,
    const flatbuffers::Vector<float>* r,
    const flatbuffers::Vector<float>* price,
    char type,
    std::vector<float>& out_iv
);
#endif

/**
 * Scalar implementation of implied volatility calculation using Brent's method.
 */
void compute_iv_scalar(
    const float& S,
    const flatbuffers::Vector<float>* K,
    const float& T,
    const flatbuffers::Vector<float>* r,
    const flatbuffers::Vector<float>* price,
    char type,
    std::vector<float>& out_iv
);

/**
 * Main entry point for implied volatility calculation.
 * Will use AVX-512 implementation if available, otherwise falls back to scalar implementation.
 */
void compute_iv(
    const float& S,
    const flatbuffers::Vector<float>* K,
    const float& T,
    const flatbuffers::Vector<float>* r,
    const flatbuffers::Vector<float>* price,
    char type,
    std::vector<float>& out_iv
);

}
