#pragma once

#include "circular_index.cuh"

template<typename T>
class Plant {
public:
	static constexpr unsigned memorySize = 4;

	static constexpr T k   = 1.53;
	static constexpr T tau = 0.0414;

	__device__ void update(const T controlSignal[memorySize], T ts, const unsigned indexArray[memorySize])
	{
		auto& u = controlSignal;

		const auto i0	 = indexArray[0];
		const auto i1	 = indexArray[1];
		const auto i2	 = indexArray[2];
		const auto iNext = indexArray[3];

		y_[i0] = (k * ts * ts * u[i0] + 2 * k * ts * ts * u[i1] + k * ts * ts * u[i2] + 8 * tau * y_[i1]
				  - (4 * tau - 2 * ts) * y_[i2])
			   / (4 * tau + 2 * ts);

		y_[iNext] = (k * ts * ts * u[i0] + 2 * k * ts * ts * u[i1] + k * ts * ts * u[i2] + 8 * tau * y_[i0]
					 - (4 * tau - 2 * ts) * y_[i1])
				  / (4 * tau + 2 * ts);
	}

	[[nodiscard]] __device__ auto getOutput() const { return y_; }

private:
	T y_[memorySize]{};
};
