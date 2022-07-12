string(TOLOWER ${ProjectName} ExecutableName)

find_package(TBB REQUIRED)

file(GLOB_RECURSE "${ExecutableName}_sources" RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS
	${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/*.cu
	${CMAKE_CURRENT_SOURCE_DIR}/*.cuh
)
list(REMOVE_ITEM ${ExecutableName}_sources pch.h)
message(STATUS "${ExecutableName}_sources: ${${ExecutableName}_sources}")

add_executable(${ExecutableName} ${${ExecutableName}_sources})
target_link_libraries(${ExecutableName} PRIVATE TBB::tbb)
target_include_directories(${ExecutableName} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(supportsIPO)
	set_target_properties(${ExecutableName} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif(supportsIPO)

target_precompile_headers(${ExecutableName} PRIVATE pch.h)

if (CMAKE_CXX_COMPILER_ID MATCHES "(Clang|GNU)")
	target_compile_options(${ExecutableName} PRIVATE
	"$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-fno-exceptions;$<$<CONFIG:RELEASE>:-O3;-fno-rtti>>")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(${ExecutableName} PRIVATE
		"$<$<COMPILE_LANGUAGE:CXX>:/Wall;/MP;/Oi;/GL;$<$<CONFIG:RELEASE>:/Ot;/GR->>")
	set_target_properties(${ExecutableName} PROPERTIES MSVC_RUNTIME_LIBRARY
		"MultiThreaded$<$<CONFIG:Debug>:Debug>")

	# Disable exceptions: https://cmake.org/pipermail/cmake/2010-December/041641.html
	if(CMAKE_CXX_FLAGS MATCHES "/EHsc ")
		string(REPLACE "/EHsc" "/EHs-c-" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	endif(CMAKE_CXX_FLAGS MATCHES "/EHsc ")

	if(CMAKE_CXX_FLAGS MATCHES "/EHsc$")
		string(REPLACE "/EHsc" "/EHs-c-" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	endif(CMAKE_CXX_FLAGS MATCHES "/EHsc$")
endif()

set_target_properties(${ExecutableName} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(MSVC)
	set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${ExecutableName})
endif(MSVC)