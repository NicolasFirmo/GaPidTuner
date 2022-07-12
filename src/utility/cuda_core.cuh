#pragma once

#ifdef NDEBUG
static constexpr bool cudaAssertionOn = false;
#else
static constexpr bool cudaAssertionOn = true;
#endif // NDEBUG

constexpr void cudaCall(cudaError_t err) {
	if constexpr (cudaAssertionOn) {
		if (err != cudaSuccess) {
			printf("%s at %s:%d: %s\n", cudaGetErrorName(err), __FILE__, __LINE__,
				   cudaGetErrorString(err));
			__debugbreak();
		}
	}
}
constexpr void afterKernelCall() {
	if constexpr (cudaAssertionOn) {
		cudaCall(cudaGetLastError());
		cudaCall(cudaDeviceSynchronize());
	}
}
