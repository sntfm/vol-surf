#pragma once

#include <string>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "flatbuffers/flatbuffers.h"
#include "OptionData_generated.h"  // Will be generated from the .fbs file

class OptionDataReader {
public:
    OptionDataReader(const std::string& shm_name = "option_chains");
    ~OptionDataReader();

    // Get the root of the FlatBuffer data
    const OptionData::OptionChainList* getOptionChains();
    
    // Refresh the view of shared memory (call this to get updated data)
    bool refresh();

private:
    std::string shm_name_;
    int shm_fd_;
    void* mapped_memory_;
    size_t mapped_size_;
    
    void cleanup();
    bool map_shared_memory();
}; 