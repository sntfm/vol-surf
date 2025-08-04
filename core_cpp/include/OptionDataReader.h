#pragma once

#include <string>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "flatbuffers/flatbuffers.h"
#include "OptionData_generated.h"  // Generated from the .fbs file

class OptionDataReader {
public:
    OptionDataReader(const std::string& shm_name = "option_chains");
    ~OptionDataReader();

    const OptionData::OptionChainList* getOptionChains(); // Gets the root of the FlatBuffer data
    bool refresh(); // Refreshes the shared memory mapping

private:
    std::string shm_name_;
    int shm_fd_;
    void* mapped_memory_
    size_t mapped_size_;
    
    void cleanup();
    bool map_shared_memory();
}; 