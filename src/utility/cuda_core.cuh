#pragma once
#include <cstdio>

#ifdef NDEBUG
static constexpr bool cudaAssertionOn = false;
#else
static constexpr bool cudaAssertionOn = true;
#endif // NDEBUG


void cudaCall(cudaError_t err) {
	if constexpr (cudaAssertionOn) {
		if (err != cudaSuccess) {
			printf("%s at %s:%d: %s\n", cudaGetErrorName(err), __FILE__, __LINE__,
				   cudaGetErrorString(err));
			debugBreak();
		}
	}
}
void afterKernelCall() {
	if constexpr (cudaAssertionOn) {
		cudaCall(cudaGetLastError());
		cudaCall(cudaDeviceSynchronize());
	}
}
