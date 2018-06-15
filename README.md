# Exercises using CUDA
	A few exercises to getting started into CUDA programming.

#### List of Exercises:

- Vector Add
- Matrix Multiplication

## Requirements

- CUDA v9.0 or later is mandatory.
- A C++11-capable compiler compatible with your version of CUDA.
- CMake v3.1 or later
- CUB Library
- CUDA_API_WRAPPERS Library

## How to Run

- Run the compile.sh script. It will do all the work.

#### Note: In case of error during the build of cuda-api_wrappers, the line include_directories(../cub/cub) should be included in its CMakeLists.txt file.

## Author

Diego Gomes Tomé - tome@cwi.nl