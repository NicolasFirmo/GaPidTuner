set(projectName GaPidTuner)

file(GLOB_RECURSE "${projectName}_sources" RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS
	${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/*.cu
	${CMAKE_CURRENT_SOURCE_DIR}/*.cuh
)
list(REMOVE_ITEM ${projectName}_sources pch.h)
message(STATUS "${projectName}_sources: ${${projectName}_sources}")

add_executable(${projectName} ${${projectName}_sources})
target_include_directories(${projectName} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(supportsIPO)
	set_target_properties(${projectName} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif(supportsIPO)

if(MSVC)
	target_compile_options(${projectName} PRIVATE
		"$<$<COMPILE_LANGUAGE:CXX>:/Wall;/MP;/Oi;/GL;$<$<CONFIG:RELEASE>:/Ot;/GR->>")
	set_target_properties(${projectName} PROPERTIES MSVC_RUNTIME_LIBRARY
		"MultiThreaded$<$<CONFIG:Debug>:Debug>")

	# Disable exceptions: https://cmake.org/pipermail/cmake/2010-December/041641.html
	if(CMAKE_CXX_FLAGS MATCHES "/EHsc ")
		string(REPLACE "/EHsc" "/EHs-c-" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	endif(CMAKE_CXX_FLAGS MATCHES "/EHsc ")

	if(CMAKE_CXX_FLAGS MATCHES "/EHsc$")
		string(REPLACE "/EHsc" "/EHs-c-" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	endif(CMAKE_CXX_FLAGS MATCHES "/EHsc$")
endif(MSVC)

target_precompile_headers(${projectName} PRIVATE pch.h)

set_target_properties(${projectName} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(MSVC)
	set_property(DIRECTORY ${PROJECT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${projectName})
endif(MSVC)