# Vol-Surf: High-Performance Option Volatility Surface Builder

A high-performance library for fetching option chain data, calculating implied volatilities, and fitting volatility surfaces using the SVI (Stochastic Volatility Inspired) parameterization. The project combines Python data fetching with C++ computational engines for maximum performance.

## Features

- **Fast Data Collection**: Python-based option chain data fetching from Yahoo Finance
- **High-Performance Processing**: 
  - C++ computational engine with SIMD optimizations
  - OpenMP parallel processing support
  - Efficient implied volatility solver using Jaeckel's method
  - SVI (Stochastic Volatility Inspired) surface fitting
- **Cross-Language Integration**: FlatBuffers-based efficient data exchange between Python and C++
- **Real-time Processing**: Shared memory mapping for fast data access

## Project Structure

```
vol-surf/
├── conn_py/               # Python connector components
│   ├── connector.py       # Yahoo Finance data fetcher
│   ├── flatbuf.py         # FlatBuffers serialization
│   └── OptionData/        # Python option data structures
├── core_cpp/              # C++ computational engine
│   ├── include/           # Header files
│   └── src/               # Implementation files
└── schemas/               # FlatBuffers schema definitions
```

## Installation

### Python Components
```bash
cd conn_py
./install_py.sh
```

### C++ Components
```bash
cd core_cpp
./install_libs.sh
mkdir build && cd build
cmake -build .
```

## Technical Details

### Volatility Surface Fitting

The project uses the SVI (Stochastic Volatility Inspired) parameterization for fitting volatility surfaces with the following parameters:
- `a`: Level
- `b`: Angle
- `rho`: Correlation (-1 to 1)
- `m`: Translation
- `sigma`: Smoothness

### Performance Optimizations

- SIMD vectorization for x86_64 architectures using AVX-512 for computing implied volatilities with Newton-Raphson method
- Automatic scalar fallback using Jaeckel's approximation for architectures not supporting AVX-512
- OpenMP parallel processing for multi-threaded computation
- Efficient memory mapping for data exchange

### Data Flow

1. Python connector fetches option chain data from Yahoo Finance
2. Data is serialized using FlatBuffers
3. C++ engine reads data via shared memory
4. Parallel processing of implied volatilities and surface fitting
5. Results are made available for analysis and visualization

## Dependencies

### Python
- yfinance
- numpy
- flatbuffers
- asyncio

### C++
- OpenMP
- FlatBuffers
- CMake (build system)
- Intel SVML (for vectorized math operations)
