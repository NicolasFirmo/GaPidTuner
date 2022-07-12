cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED  True)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 50)
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

get_filename_component(RootFolderName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${RootFolderName} LANGUAGES CXX CUDA)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

include(CheckIPOSupported)
check_ipo_supported(RESULT supportsIPO OUTPUT error)
if(supportsIPO)
	message(STATUS "IPO / LTO is supported")
else()
	message(STATUS "IPO / LTO is not supported: <${error}>")
endif()

if(NOT DEFINED BUILD_SHARED_LIBS)
	set(BUILD_SHARED_LIBS OFF)
endif(NOT DEFINED BUILD_SHARED_LIBS)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Get directories names: https://stackoverflow.com/a/7788165
file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/*
)
list(REMOVE_ITEM children scripts build .git .vs  .vscode)
set(subdirs "")
foreach(child ${children})
	if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child})
		list(APPEND subdirs ${child})
	endif()
endforeach()

foreach(subdir ${subdirs})
	add_subdirectory(${subdir})
endforeach()