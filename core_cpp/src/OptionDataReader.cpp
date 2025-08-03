#include "OptionDataReader.h"
#include <iostream>

OptionDataReader::OptionDataReader(const std::string& shm_name)
    : shm_name_(shm_name)
    , shm_fd_(-1)
    , mapped_memory_(nullptr)
    , mapped_size_(0) {
    map_shared_memory();
}

OptionDataReader::~OptionDataReader() {
    cleanup();
}

void OptionDataReader::cleanup() {
    if (mapped_memory_ != nullptr) {
        munmap(mapped_memory_, mapped_size_);
        mapped_memory_ = nullptr;
    }
    if (shm_fd_ != -1) {
        close(shm_fd_);
        shm_fd_ = -1;
    }
}

bool OptionDataReader::map_shared_memory() {
    cleanup();  // Clean up any existing mapping

    // Try both with and without leading slash
    std::string names[] = {shm_name_, "/" + shm_name_};
    
    for (const auto& name : names) {
        // Open shared memory
        shm_fd_ = shm_open(name.c_str(), O_RDONLY, 0666);
        if (shm_fd_ != -1) {
            // Successfully opened, get the size
            struct stat sb;
            if (fstat(shm_fd_, &sb) == -1) {
                std::cerr << "Failed to get shared memory size: " << strerror(errno) << std::endl;
                cleanup();
                continue;
            }
            mapped_size_ = sb.st_size;

            // Map the shared memory
            mapped_memory_ = mmap(nullptr, mapped_size_, PROT_READ, MAP_SHARED, shm_fd_, 0);
            if (mapped_memory_ != MAP_FAILED) {
                return true;  // Successfully mapped
            }
            
            std::cerr << "Failed to map shared memory: " << strerror(errno) << std::endl;
            mapped_memory_ = nullptr;
            cleanup();
        }
    }

    std::cerr << "Failed to open shared memory: " << strerror(errno) << std::endl;
    return false;
}

bool OptionDataReader::refresh() {
    return map_shared_memory();
}

const OptionData::OptionChainList* OptionDataReader::getOptionChains() {
    if (!mapped_memory_) {
        return nullptr;
    }
    
    // Verify the buffer and get root
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(mapped_memory_), mapped_size_);
    if (!OptionData::VerifyOptionChainListBuffer(verifier)) {
        std::cerr << "Invalid FlatBuffer data in shared memory" << std::endl;
        return nullptr;
    }
    
    return OptionData::GetOptionChainList(mapped_memory_);
} 