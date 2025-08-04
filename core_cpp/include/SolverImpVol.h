#pragma once

#include <vector>
#include "flatbuffers/flatbuffers.h"

namespace SolverImpVol {

#if defined(__x86_64__)
// AVX-512 implementation of IV calculation using Newton-Raphson method
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

// Scalar implementation of IV calculation using Jaeckel's approximation
void compute_iv_scalar(
    const float& S,
    const flatbuffers::Vector<float>* K,
    const float& T,
    const flatbuffers::Vector<float>* r,
    const flatbuffers::Vector<float>* price,
    char type,
    std::vector<float>& out_iv
);

// Main entry point for IV calculation, will use AVX-512 if available, otherwise falls back to scalar
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
