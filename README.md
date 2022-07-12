# PID controler tuner using GA

## Requirements

- CMake
- CUDA Toolkit

## Building

1. Generating build files
    
    1.1. Manual

    - Create a build folder inside the root directory and then run `cmake ..` from there

    1.2. Script

    - Double click the [win_generate.bat](scripts/win_generate.bat) file inside the [scripts](scripts) folder

2. Generating the executable
    
    2.1. Windows

    - Open the generated .sln file inside the build directory with Visual Studio
    - Build the solution

    2.2. Unix

    - Run your generator command inside the build folder

        For example, to build with Unix Makefiles use `make .`

