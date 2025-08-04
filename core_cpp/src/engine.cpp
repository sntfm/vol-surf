#include "OptionDataReader.h"
#include "SolverImpVol.h"
#include "SurfaceFitter.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <omp.h>

void printOptionChain(const OptionData::OptionChain* chain) {
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

void profile_reads(OptionDataReader& reader, int num_reads = 1000) {
    std::cout << "\nProfiling " << num_reads << " consecutive reads...\n";
    
    // Warm-up read
    reader.getOptionChains();
    
    // Profile multiple reads
    auto start = std::chrono::high_resolution_clock::now();
    size_t total_chains = 0;
    
    for (int i = 0; i < num_reads; ++i) {
        const auto* chains = reader.getOptionChains();
        if (chains && chains->chains()) {
            total_chains += chains->chains()->size();
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_read_time = static_cast<double>(duration.count()) / num_reads;
    std::cout << "Average read time: " << avg_read_time << " µs\n";
    std::cout << "Total chains processed: " << total_chains << "\n";
}

int main() {
    // Measure initial mapping time
    auto start = std::chrono::high_resolution_clock::now();
    OptionDataReader reader("option_chains");
    auto end = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Initial shared memory mapping took: " << init_duration.count() << " µs\n";
    
    // Measure first read and verification time
    start = std::chrono::high_resolution_clock::now();
    const OptionData::OptionChainList* chains = reader.getOptionChains();
    end = std::chrono::high_resolution_clock::now();
    auto first_read_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "First read and verification took: " << first_read_duration.count() << " µs\n";
    
    if (!chains) {
        std::cerr << "Failed to read option chains from shared memory" << std::endl;
        return 1;
    }
    
    std::cout << "\nFound " << chains->chains()->size() << " option chain(s)\n";
    
    // Process all chains (all expiries)
    auto total_start = std::chrono::high_resolution_clock::now();
    int total_options = 0;
    int successful_fits = 0;
    
    std::cout << "\nProcessing all option chains:\n";
    std::cout << std::string(100, '=') << "\n";
    
    // Process chains in parallel
    #pragma omp parallel for schedule(dynamic) reduction(+:total_options,successful_fits)
    for (size_t chain_idx = 0; chain_idx < chains->chains()->size(); ++chain_idx) {
        const auto* chain = chains->chains()->Get(chain_idx);
        
        // Skip if no data
        if (!chain->calls_strike() || chain->calls_strike()->size() == 0) {
            continue;
        }
        
        float spot_price = chain->spot_price();
        float tau_years = chain->tau_years();
        auto* rfr = chain->rfr();
        
        // Get option data
        auto* calls_strike = chain->calls_strike();
        auto* calls_ask = chain->calls_ask();
        
        // Prepare moneyness
        std::vector<float> moneyness(calls_strike->size());
        float forward = spot_price * std::exp(rfr->Get(0) * tau_years);
        for (size_t i = 0; i < calls_strike->size(); ++i) {
            moneyness[i] = std::log(calls_strike->Get(i) / forward);
        }
        
        // Compute IV for this chain
        auto chain_iv_start = std::chrono::high_resolution_clock::now();
        std::vector<float> ivs;
        SolverImpVol::compute_iv(spot_price, calls_strike, tau_years, rfr, calls_ask, 'C', ivs);
        auto chain_iv_end = std::chrono::high_resolution_clock::now();
        auto chain_iv_duration = std::chrono::duration_cast<std::chrono::microseconds>(chain_iv_end - chain_iv_start);
        
        // SVI fitting for this chain - optimized for maximum speed
        SurfaceFitter::SVI::Fitter fitter;
        SurfaceFitter::SVI::Params params;
        
        auto chain_svi_start = std::chrono::high_resolution_clock::now();
        bool fit_success = false;
        
        // Use clean SVI fitting interface
        if (moneyness.size() > 2 && ivs.size() > 2 && moneyness.size() == ivs.size()) {
            fit_success = fitter.fit_slice(moneyness, ivs, params, 500.0f);  // 500 microsecond budget
        }
        
        auto chain_svi_end = std::chrono::high_resolution_clock::now();
        auto chain_svi_duration = std::chrono::duration_cast<std::chrono::microseconds>(chain_svi_end - chain_svi_start);
        
        total_options += calls_strike->size();
        if (fit_success) successful_fits++;
        
        // Thread-safe output (critical section)
        #pragma omp critical
        {
            std::cout << "\nChain " << (chain_idx + 1) << "/" << chains->chains()->size();
            if (chain->expiration()) {
                std::cout << " (Expiry: " << chain->expiration()->c_str() << ")";
            }
            std::cout << ":\n";
            std::cout << "  Time to expiry: " << std::fixed << std::setprecision(4) << tau_years << " years\n";
            std::cout << "  Strikes: " << calls_strike->size() << "\n";
            std::cout << "  IV computation: " << chain_iv_duration.count() << " µs ("
                      << std::setprecision(2) << static_cast<double>(chain_iv_duration.count()) / calls_strike->size() 
                      << " µs/option)\n";
            std::cout << "  SVI fitting: " << chain_svi_duration.count() << " µs\n";
            std::cout << "  Fit success: " << (fit_success ? "YES" : "NO") << "\n";
            
            if (fit_success) {
                std::cout << "  SVI params: a=" << std::setprecision(4) << params.a 
                          << " b=" << params.b << " rho=" << params.rho 
                          << " m=" << params.m << " sigma=" << params.sigma << "\n";
            }
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    // Print detailed parameters for chain 0
    if (chains->chains()->size() > 0) {
        const auto* chain0 = chains->chains()->Get(0);
        std::cout << std::string(100, '=') << "\n";
        std::cout << "DETAILED PARAMETERS FOR CHAIN 0:\n";
        std::cout << std::string(100, '=') << "\n";
        
        if (chain0->expiration()) {
            std::cout << "Expiration: " << chain0->expiration()->c_str() << "\n";
        }
        std::cout << "Spot Price: " << std::fixed << std::setprecision(4) << chain0->spot_price() << "\n";
        std::cout << "Time to expiry: " << chain0->tau_years() << " years\n";
        std::cout << "Risk-free rate: " << chain0->rfr()->Get(0) << "\n";
        
        float spot_price = chain0->spot_price();
        float tau_years = chain0->tau_years();
        auto* rfr = chain0->rfr();
        auto* calls_strike = chain0->calls_strike();
        auto* calls_ask = chain0->calls_ask();
        auto* calls_bid = chain0->calls_bid();
        
        // Compute forward price
        float forward = spot_price * std::exp(rfr->Get(0) * tau_years);
        std::cout << "Forward Price: " << std::setprecision(4) << forward << "\n\n";
        
        // Prepare moneyness and compute IVs
        std::vector<float> moneyness(calls_strike->size());
        for (size_t i = 0; i < calls_strike->size(); ++i) {
            moneyness[i] = std::log(calls_strike->Get(i) / forward);
        }
        
        std::vector<float> ivs;
        SolverImpVol::compute_iv(spot_price, calls_strike, tau_years, rfr, calls_ask, 'C', ivs);
        
        // Print option data table
        std::cout << "OPTION DATA:\n";
        std::cout << std::setw(8) << "Strike" << std::setw(8) << "Bid" << std::setw(8) << "Ask" 
                  << std::setw(12) << "Moneyness" << std::setw(8) << "IV" << "\n";
        std::cout << std::string(50, '-') << "\n";
        
        for (size_t i = 0; i < calls_strike->size(); ++i) {
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(8) << calls_strike->Get(i)
                      << std::setw(8) << calls_bid->Get(i)
                      << std::setw(8) << calls_ask->Get(i)
                      << std::setprecision(4)
                      << std::setw(12) << moneyness[i];
            
            if (i < ivs.size()) {
                std::cout << std::setw(8) << ivs[i];
            } else {
                std::cout << std::setw(8) << "N/A";
            }
            std::cout << "\n";
        }
        
        // Fit SVI and print parameters
        std::cout << "\nSVI FITTING:\n";
        SurfaceFitter::SVI::Fitter fitter;
        SurfaceFitter::SVI::Params params;
        
        if (moneyness.size() > 2 && ivs.size() > 2 && moneyness.size() == ivs.size()) {
            bool fit_success = fitter.fit_slice(moneyness, ivs, params, 1000.0f);  // More time for detailed analysis
            std::cout << "Fit successful: " << (fit_success ? "YES" : "NO") << "\n";
            
            if (fit_success) {
                std::cout << "SVI Parameters:\n";
                std::cout << "  a (intercept): " << std::setprecision(6) << params.a << "\n";
                std::cout << "  b (angle): " << params.b << "\n";
                std::cout << "  rho (correlation): " << params.rho << "\n";
                std::cout << "  m (translation): " << params.m << "\n";
                std::cout << "  sigma (vol of vol): " << params.sigma << "\n";
            }
        } else {
            std::cout << "Insufficient data for SVI fitting\n";
        }
        std::cout << "\n";
    }
    
    std::cout << std::string(100, '=') << "\n";
    std::cout << "SUMMARY:\n";
    std::cout << "  Total chains processed: " << chains->chains()->size() << "\n";
    std::cout << "  Total options processed: " << total_options << "\n";
    std::cout << "  Successful SVI fits: " << successful_fits << "/" << chains->chains()->size() 
              << " (" << std::setprecision(1) << (100.0 * successful_fits / chains->chains()->size()) << "%)\n";
    std::cout << "  Total processing time: " << total_duration.count() << " µs\n";
    std::cout << "  Average time per chain: " 
              << std::setprecision(2) << static_cast<double>(total_duration.count()) / chains->chains()->size() << " µs\n";
    std::cout << "  Average time per option: " 
              << std::setprecision(2) << static_cast<double>(total_duration.count()) / total_options << " µs\n";
    std::cout << "  OpenMP threads used: " << omp_get_max_threads() << "\n";
    
    return 0;
} 