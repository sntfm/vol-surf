cmake_minimum_required(VERSION 3.10)

# Force CMake to use GCC on macOS
if(APPLE)
    set(CMAKE_C_COMPILER "/opt/homebrew/bin/gcc-14" CACHE STRING "" FORCE)
    set(CMAKE_CXX_COMPILER "/opt/homebrew/bin/g++-14" CACHE STRING "" FORCE)
    
    # Configure OpenMP for GCC
    set(OpenMP_C_FLAGS "-fopenmp")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_C_LIB_NAMES "gomp")
    set(OpenMP_CXX_LIB_NAMES "gomp")
    set(OpenMP_gomp_LIBRARY "/opt/homebrew/lib/gcc/14/libgomp.dylib")
endif()

project(OptionDataReader)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Check for AVX-512 support
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
    #include <immintrin.h>
    int main() {
        __m512 a = _mm512_set1_ps(1.0f);
        return 0;
    }
" HAS_AVX512)

if(HAS_AVX512)
    message(STATUS "AVX-512 support detected")
    add_definitions(-DHAS_AVX512)
    if(MSVC)
        add_compile_options(/arch:AVX512)
    else()
        add_compile_options(-mavx512f -mavx512dq)
    endif()
else()
    message(STATUS "No AVX-512 support, using fallback implementation")
endif()

# Set compiler flags for optimization
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    
    if(HAS_AVX512)
        message(STATUS "AVX-512 support detected")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512bw -mavx512vl")
        add_definitions(-DHAS_AVX512)
    else()
        message(STATUS "No AVX-512 support, using fallback implementation")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
        endif()
    endif()
endif()

# Set include directories explicitly
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /opt/homebrew/include
    /opt/homebrew/Cellar/flatbuffers/25.2.10/include
)

# Find FlatBuffers
find_path(FLATBUFFERS_INCLUDE_DIR NAMES flatbuffers/flatbuffers.h
    PATHS 
        /opt/homebrew/include
        /opt/homebrew/Cellar/flatbuffers/25.2.10/include
    REQUIRED)

# Generate FlatBuffers code
find_program(FLATC flatc REQUIRED)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/OptionData_generated.h
    COMMAND ${FLATC} --cpp -o ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../schemas/OptionData.fbs
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../schemas/OptionData.fbs
    COMMENT "Generating FlatBuffers C++ header"
)

# Create a custom target for the generated header
add_custom_target(generate_flatbuffers
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/OptionData_generated.h
)

# Add OptionDataReader library
add_library(OptionDataReader
    src/OptionDataReader.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/OptionData_generated.h
)

# Add SolverIV library
add_library(SolverIV
    src/SolverIV.cpp
)

# Add SVIFitter library
add_library(SVIFitter
    src/SVIFitter.cpp
)

# Add OpenMP to SolverIV
target_link_libraries(SolverIV PRIVATE OpenMP::OpenMP_CXX)

# Add include directories for the OptionDataReader library
target_include_directories(OptionDataReader
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
        ${FLATBUFFERS_INCLUDE_DIR}
)

# Add include directories for the SolverIV library
target_include_directories(SolverIV
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add include directories for the SVIFitter library
target_include_directories(SVIFitter
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add dependencies
add_dependencies(OptionDataReader generate_flatbuffers)

# Add example executable
add_executable(example_reader src/example_reader.cpp)
target_link_libraries(example_reader 
    PRIVATE 
        OptionDataReader
        SolverIV
        SVIFitter
)
target_include_directories(example_reader
    PRIVATE
        ${FLATBUFFERS_INCLUDE_DIR}
) 