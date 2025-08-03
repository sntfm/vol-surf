#include "OptionDataReader.h"
#include "SolverIV.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

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
    using namespace std::chrono;
    
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
    auto duration = duration_cast<std::chrono::microseconds>(end - start);
    
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
    

    // // Optional: print first chain as sample data
    // if (chains->chains()->size() > 0) {
    //     std::cout << "\nSample data (first chain):\n";
    //     printOptionChain(chains->chains()->Get(0));
    // }

    // // Profile reads
    // profile_reads(reader);

    
    // std::cout << "\nCALLS:" << std::endl;
    // std::cout << std::setw(10) << "Strike" << std::setw(10) << "Bid" << std::setw(10) << "Ask" << std::endl;
    // for (size_t i = 0; i < chain->calls_strike()->size(); ++i) {
    //     std::cout << std::fixed << std::setprecision(2)
    //               << std::setw(10) << chain->calls_strike()->Get(i)
    //               << std::setw(10) << chain->calls_bid()->Get(i)
    //               << std::setw(10) << chain->calls_ask()->Get(i) << std::endl;

    // Print first chain
    if (chains->chains()->size() > 0) {
        const auto* chain = chains->chains()->Get(0);

        float spot_price = chain->spot_price();
        float tau_years = chain->tau_years();
        const auto* rfr = chain->rfr();
        auto* calls_strike = chain->calls_strike();
        auto* calls_bid = chain->calls_bid();
        auto* calls_ask = chain->calls_ask();
        auto* puts_strike = chain->puts_strike();
        auto* puts_bid = chain->puts_bid();
        auto* puts_ask = chain->puts_ask();

        std::vector<float> ivs;
        iv::compute_iv(spot_price, calls_strike, tau_years, rfr, calls_ask, 'C', ivs);
        
        // Print header
        std::cout << chain->expiration()->str() << std::endl;
        std::cout << std::setw(10) << "Strike" 
                  << std::setw(10) << "Ask" 
                  << std::setw(10) << "Spot" 
                  << std::setw(10) << "T" 
                  << std::setw(10) << "RFR"
                  << std::setw(10) << "IV" << std::endl;
        
        // Print data for each strike
        for (size_t i = 0; i < calls_strike->size(); ++i) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(10) << calls_strike->Get(i)
                      << std::setw(10) << calls_ask->Get(i)
                      << std::setw(10) << spot_price
                      << std::setw(10) << tau_years
                      << std::setw(10) << rfr->Get(i)
                      << std::setw(10) << ivs[i] << std::endl;
        }
    }
    
    return 0;
} 