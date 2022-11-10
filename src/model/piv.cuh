#pragma once
#include "circular_index.cuh"
#include "utility/redomain.cuh"

#include <cmath>

static constexpr float pi = 3.14159265F;

template<typename T, unsigned MemorySize>
class PIV {
public:
	__device__ PIV(T kp, T ki, T kv) : kp_(kp), ki_(ki), kv_(kv) {}

	__device__ void update(const T error[MemorySize], const T measuredY[MemorySize], T ts,
						   const unsigned indexArray[MemorySize]) {
		const auto i0 = indexArray[0];
		const auto i1 = indexArray[1];

		const T proportional = error[i0];
		const T newIntegral = integral_ + ((error[i0] + error[i1]) / 2) * ts;
		const T velocity = (quantize(measuredY[i0], 2 * pi, encoderResolution)
							- quantize(measuredY[i1], 2 * pi, encoderResolution))
						 / ts;
		const T u = kp_ * proportional - kv_ * velocity + ki_ * newIntegral;
		v_[i0] = saturate(u, {-saturationValue, saturationValue});

		if (!isGoingWindup(error[i0], u, v_[i0]))
			integral_ = newIntegral;
	}

	[[nodiscard]] __device__ auto getControlSignal() const { return v_; }

	static constexpr T saturationValue = 6.0;
	static constexpr int encoderResolution = 4096;

private:
	__host__ __device__ constexpr T isGoingWindup(const T error, const T u, const T v) {
		return u != v && u * error > T{0};
	}

	T kp_;
	T ki_;
	T kv_;

	T v_[MemorySize]{};
	T integral_{0};
};