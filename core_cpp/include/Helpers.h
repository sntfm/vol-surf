#pragma once

#include "OptionData_generated.h"
#include "SurfaceFitter.h"

#include <chrono>
#include <iostream>
#include <string>
#include <iomanip>

// #define ENABLE_PROFILING // comment this out to disable profiling

#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define CONCAT_IMPL(a, b) a##b

#ifdef ENABLE_PROFILING
    #define PROFILE_SCOPE(name) Helpers::ScopedTimer CONCAT(timer_, __LINE__)((name))
    #define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#else
    #define PROFILE_SCOPE(name) ((void)0)
    #define PROFILE_FUNCTION() ((void)0)
#endif

namespace Helpers {

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name)
        : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        std::cout << name_ << " took " << us << " µs\n";
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

inline void printOptionChain(const OptionData::OptionChain* chain) {
    std::cout << "Expiration: " << chain->expiration()->str() << std::endl;
    std::cout << "Spot Price: " << chain->spot_price() << std::endl;
    
    std::cout << "\nCALLS:" << std::endl;
    std::cout << std::setw(10) << "Strike" << std::setw(10) << "Bid" << std::setw(10) << "Ask" << std::endl;
    for (size_t i = 0; i < chain->calls_strike()->size(); ++i) {
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << chain->calls_strike()->Get(i)
                  << std::setw(10) << chain->calls_bid()->Get(i)
                  << std::setw(10) << chain->calls_ask()->Get(i) << std::endl;
    }
    
    std::cout << "\nPUTS:" << std::endl;
    std::cout << std::setw(10) << "Strike" << std::setw(10) << "Bid" << std::setw(10) << "Ask" << std::endl;
    for (size_t i = 0; i < chain->puts_strike()->size(); ++i) {
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << chain->puts_strike()->Get(i)
                  << std::setw(10) << chain->puts_bid()->Get(i)
                  << std::setw(10) << chain->puts_ask()->Get(i) << std::endl;
    }
    std::cout << "\n";
}

inline void printSurfaceParams(const SurfaceFitter::SVI::Params& params) {
    std::cout << "SVI Parameters:\n";
    std::cout << "  a (intercept): " << std::setprecision(6) << params.a << "\n";
    std::cout << "  b (angle): " << params.b << "\n";
    std::cout << "  rho (correlation): " << params.rho << "\n";
    std::cout << "  m (translation): " << params.m << "\n";
    std::cout << "  sigma (vol of vol): " << params.sigma << "\n";
}

} // namespace Helpers