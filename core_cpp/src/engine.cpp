#include "OptionDataReader.h"
#include "SolverImpVol.h"
#include "SurfaceFitter.h"
#include "Helpers.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <omp.h>

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
            fit_success = fitter.fit_slice(moneyness, ivs, params, 500.0f);
        }
        
        auto chain_svi_end = std::chrono::high_resolution_clock::now();
        auto chain_svi_duration = std::chrono::duration_cast<std::chrono::microseconds>(chain_svi_end - chain_svi_start);
        
        total_options += calls_strike->size();
        if (fit_success) successful_fits++;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    std::cout << "SUMMARY:\n";
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
    std::cout << "  OpenMP threads used: " << omp_get_max_threads() << "\n";

    return 0;
} 