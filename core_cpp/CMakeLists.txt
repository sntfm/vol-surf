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

# Option to automatically install dependencies
option(AUTO_INSTALL_DEPENDENCIES "Automatically install missing dependencies" OFF)

# Function to check and install dependencies
function(check_and_install_dependencies)
    set(DEPENDENCIES_MISSING FALSE)
    
    # Check for required programs
    find_program(GCC_FOUND gcc)
    find_program(GPP_FOUND g++)
    find_package(OpenMP QUIET)
    find_package(Flatbuffers QUIET)
    
    if(NOT GCC_FOUND OR NOT GPP_FOUND OR NOT OpenMP_FOUND OR NOT Flatbuffers_FOUND)
        set(DEPENDENCIES_MISSING TRUE)
    endif()
    
    if(DEPENDENCIES_MISSING)
        message(STATUS "Some dependencies are missing.")
        if(AUTO_INSTALL_DEPENDENCIES)
            message(STATUS "Attempting to install missing dependencies...")
            execute_process(
                COMMAND ${CMAKE_SOURCE_DIR}/install_libs.sh
                RESULT_VARIABLE INSTALL_RESULT
            )
            if(NOT INSTALL_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to install dependencies. Please install them manually.")
            endif()
        else()
            message(STATUS "Run CMake with -DAUTO_INSTALL_DEPENDENCIES=ON to automatically install dependencies")
            message(STATUS "Or run /install_libs.sh manually")
        endif()
    endif()
endfunction()

# Check dependencies at configure time
check_and_install_dependencies()

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
        // Check AVX-512F (Foundation)
        __m512 a = _mm512_set1_ps(1.0f);
        __mmask16 mask = _mm512_cmp_ps_mask(a, a, _CMP_LT_OS);
        
        // Check FMA
        __m512 fma = _mm512_fmadd_ps(a, a, a);
        
        // Check masked operations
        __m512 masked = _mm512_mask_sub_ps(a, mask, a, a);
        
        // Check AVX-512DQ (Double/Quad)
        __m512d d = _mm512_set1_pd(1.0);
        
        return 0;
    }
" HAS_AVX512)

if(HAS_AVX512)
    message(STATUS "AVX-512 support detected with all required features")
    add_definitions(-DHAS_AVX512)
    if(MSVC)
        add_compile_options(/arch:AVX512)
    else()
        add_compile_options(-mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma)
    endif()
else()
    message(STATUS "Full AVX-512 support not available, using fallback implementation")
endif()

# Set compiler flags for optimization
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize")
    
    if(HAS_AVX512)
        message(STATUS "AVX-512 support detected")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma")
        add_definitions(-DHAS_AVX512)
    else()
        message(STATUS "No AVX-512 support, using fallback implementation")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
        endif()
    endif()
endif()

# Find FlatBuffers using standard CMake paths
find_package(Flatbuffers QUIET)
if(NOT Flatbuffers_FOUND)
    # Try pkg-config as fallback
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(Flatbuffers QUIET flatbuffers)
    endif()
    
    # If still not found, look in some common paths
    if(NOT Flatbuffers_FOUND)
        find_path(FLATBUFFERS_INCLUDE_DIR NAMES flatbuffers/flatbuffers.h
            PATHS
                # Linux paths
                /usr/include
                /usr/local/include
                # macOS Homebrew paths
                /opt/homebrew/include
                /usr/local/opt/flatbuffers/include
                # Custom paths
                ${CMAKE_SOURCE_DIR}/third_party/flatbuffers/include
            PATH_SUFFIXES flatbuffers
        )
        if(FLATBUFFERS_INCLUDE_DIR)
            set(Flatbuffers_FOUND TRUE)
        endif()
    endif()
endif()

if(NOT Flatbuffers_FOUND)
    message(FATAL_ERROR "FlatBuffers not found. Please install FlatBuffers:\n"
                       "  Linux: sudo apt install flatbuffers-compiler libflatbuffers-dev\n"
                       "  macOS: brew install flatbuffers\n"
                       "  Or download from https://github.com/google/flatbuffers")
endif()

# Find flatc compiler
find_program(FLATC flatc
    PATHS
        # Linux paths
        /usr/bin
        /usr/local/bin
        # macOS Homebrew paths
        /opt/homebrew/bin
        /usr/local/opt/flatbuffers/bin
        # Custom paths
        ${CMAKE_SOURCE_DIR}/third_party/flatbuffers/bin
    REQUIRED
)

# Generate FlatBuffers code
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

# Add include directories for the OptionDataReader library
target_include_directories(OptionDataReader
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
        ${FLATBUFFERS_INCLUDE_DIR}
)

# Add SolverImpVol library
add_library(SolverImpVol
    src/SolverImpVol.cpp
)

# Add include directories for the SolverImpVol library
target_include_directories(SolverImpVol
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLATBUFFERS_INCLUDE_DIR}  # Add FlatBuffers headers
)

# Add OpenMP and math library to SolverImpVol
target_link_libraries(SolverImpVol 
    PRIVATE 
        OpenMP::OpenMP_CXX
        m
)

# Add SurfaceFitter library
add_library(SurfaceFitter
    src/SurfaceFitter.cpp
)

# Add include directories for the SurfaceFitter library
target_include_directories(SurfaceFitter
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add dependencies
add_dependencies(OptionDataReader generate_flatbuffers)

# Add executable
add_executable(engine src/engine.cpp)

# Add include directories for the engine executable
target_include_directories(engine
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLATBUFFERS_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(engine 
    PRIVATE 
        OptionDataReader
        SolverImpVol
        SurfaceFitter
        OpenMP::OpenMP_CXX
        m
) 