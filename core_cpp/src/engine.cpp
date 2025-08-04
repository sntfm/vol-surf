#include "OptionDataReader.h"
#include "SolverImpVol.h"
#include "SurfaceFitter.h"
#include "Helpers.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <omp.h>

// Simple storage structures
struct ChainResult {
    size_t chain_idx;
    std::string expiration;
    float spot_price;
    float tau_years;
    std::vector<float> strikes;
    std::vector<float> ivs;
    bool fit_success;
    SurfaceFitter::SVI::Params svi_params;
    float fit_quality;  // MSE
};

int main() {
    // Initial shared memory mapping
    auto start = std::chrono::high_resolution_clock::now();
    OptionDataReader reader("option_chains");
    auto end = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // First read and verification time
    start = std::chrono::high_resolution_clock::now();
    const OptionData::OptionChainList* chains = reader.getOptionChains();
    end = std::chrono::high_resolution_clock::now();
    auto first_read_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (!chains) {
        std::cerr << "Failed to read option chains from shared memory" << std::endl;
        return 1;
    }
    
    // Storage for results
    std::vector<ChainResult> results(chains->chains()->size());
    
    // Process all chains
    auto total_start = std::chrono::high_resolution_clock::now();
    int total_options = 0;
    int successful_fits = 0;
    
    #pragma omp parallel for schedule(dynamic) reduction(+:total_options,successful_fits)
    for (size_t chain_idx = 0; chain_idx < chains->chains()->size(); ++chain_idx) {
        const auto* chain = chains->chains()->Get(chain_idx);
        
        if (!chain->calls_strike() || chain->calls_strike()->size() == 0) {
            continue;
        }
        
        float spot_price = chain->spot_price();
        float tau_years = chain->tau_years();
        auto* rfr = chain->rfr();
        auto* calls_strike = chain->calls_strike();
        auto* calls_ask = chain->calls_ask();
        
        std::vector<float> moneyness(calls_strike->size());
        float forward = spot_price * std::exp(rfr->Get(0) * tau_years);
        for (size_t i = 0; i < calls_strike->size(); ++i) {
            moneyness[i] = std::log(calls_strike->Get(i) / forward);
        }
        
        // Compute implied vols for chain
        auto chain_iv_start = std::chrono::high_resolution_clock::now();
        std::vector<float> ivs;
        SolverImpVol::compute_iv(spot_price, calls_strike, tau_years, rfr, calls_ask, 'C', ivs);
        auto chain_iv_end = std::chrono::high_resolution_clock::now();
        auto chain_iv_duration = std::chrono::duration_cast<std::chrono::microseconds>(chain_iv_end - chain_iv_start);
        
        // Fit slice with SVI
        SurfaceFitter::SVI::Fitter fitter;
        SurfaceFitter::SVI::Params params;
        
        auto chain_svi_start = std::chrono::high_resolution_clock::now();
        bool fit_success = false;
        
        if (moneyness.size() > 2 && ivs.size() > 2 && moneyness.size() == ivs.size()) {
            fit_success = fitter.fit_slice(moneyness, ivs, params, 1000.0f);
        }
        
        auto chain_svi_end = std::chrono::high_resolution_clock::now();
        auto chain_svi_duration = std::chrono::duration_cast<std::chrono::microseconds>(chain_svi_end - chain_svi_start);
        
        // Calculate fit quality (MSE)
        float fit_quality = 0.0f;
        if (fit_success && ivs.size() > 0) {
            for (size_t i = 0; i < ivs.size(); ++i) {
                float model_iv = std::sqrt(std::max(0.0001f, 
                    params.a + params.b * (params.rho * moneyness[i] + 
                    std::sqrt(moneyness[i] * moneyness[i] + params.sigma * params.sigma))));
                float error = model_iv - ivs[i];
                fit_quality += error * error;
            }
            fit_quality /= ivs.size();
        }
        
        // Store results (thread-safe since each thread writes to different index)
        results[chain_idx] = {
            chain_idx,
            chain->expiration() ? chain->expiration()->c_str() : "N/A",
            spot_price,
            tau_years,
            std::vector<float>(calls_strike->begin(), calls_strike->end()),
            ivs,
            fit_success,
            params,
            fit_quality
        };
        
        total_options += calls_strike->size();
        if (fit_success) successful_fits++;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    std::cout << "\nSUMMARY:\n";
    std::cout << "  Initial shared memory mapping: " << init_duration.count() << " µs\n";
    std::cout << "  First read and verification: " << first_read_duration.count() << " µs\n";
    std::cout << "  Avg time per chain: " 
              << std::fixed << std::setprecision(1) << static_cast<double>(total_duration.count()) / chains->chains()->size() << " µs\n";
    std::cout << "  Avg time per option: " 
              << std::fixed << std::setprecision(1) << static_cast<double>(total_duration.count()) / total_options << " µs\n";

    std::cout << "  Total chains processed: " << chains->chains()->size() << "\n";
    std::cout << "  Total options processed: " << total_options << "\n";
    std::cout << "  Successful SVI fits: " << successful_fits << "/" << chains->chains()->size() 
              << " (" << std::fixed << std::setprecision(1) << (100.0 * successful_fits / chains->chains()->size()) << "%)\n";
    std::cout << "  Total processing time: " << total_duration.count() << " µs\n";
    std::cout << "  OpenMP threads used: " << omp_get_max_threads() << "\n\n";

    // Print detailed results
    std::cout << "DETAILED RESULTS:\n";
    std::cout << std::setw(5) << "Chain" 
              << std::setw(15) << "Expiration" 
              << std::setw(10) << "Spot" 
              << std::setw(8) << "TTM" 
              << std::setw(6) << "Opts" 
              << std::setw(8) << "Success" 
              << std::setw(10) << "Fit MSE" 
              << std::setw(8) << "SVI_a" 
              << std::setw(8) << "SVI_b" 
              << std::setw(8) << "SVI_rho" 
              << std::setw(8) << "SVI_m" 
              << std::setw(8) << "SVI_sig" << "\n";
    std::cout << std::string(100, '-') << "\n";
    
    for (const auto& result : results) {
        if (result.strikes.empty()) continue;  // Skip empty chains
        
        std::cout << std::setw(5) << result.chain_idx
                  << std::setw(15) << result.expiration
                  << std::setw(10) << std::fixed << std::setprecision(2) << result.spot_price
                  << std::setw(8) << std::fixed << std::setprecision(3) << result.tau_years
                  << std::setw(6) << result.strikes.size()
                  << std::setw(8) << (result.fit_success ? "YES" : "NO")
                  << std::setw(10) << std::fixed << std::setprecision(4) << result.fit_quality;
        
        if (result.fit_success) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) << result.svi_params.a
                      << std::setw(8) << std::fixed << std::setprecision(3) << result.svi_params.b
                      << std::setw(8) << std::fixed << std::setprecision(3) << result.svi_params.rho
                      << std::setw(8) << std::fixed << std::setprecision(3) << result.svi_params.m
                      << std::setw(8) << std::fixed << std::setprecision(3) << result.svi_params.sigma;
        } else {
            std::cout << std::setw(8) << "-"
                      << std::setw(8) << "-"
                      << std::setw(8) << "-"
                      << std::setw(8) << "-"
                      << std::setw(8) << "-";
        }
        std::cout << "\n";
    }

    return 0;
} 