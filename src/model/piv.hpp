#pragma once

#include "utility/accumulator.hpp"
#include "utility/redomain.hpp"

template<typename T, size_t MemorySize>
class PIV {
public:
	PIV(T kp, T ki, T kv) : kp_(kp), ki_(ki), kv_(kv) {}

	const Accumulator<T, MemorySize>& update(const Accumulator<T, MemorySize>& error,
											 const Accumulator<T, MemorySize>& measuredY, T ts)
	{
		++u_;

		const T proportional = error[0];
		const T newIntegral	 = integral_ + ((error[0] + error[1]) / 2) * ts;
		const T velocity	 = (measuredY[0] - measuredY[1]) / ts;

		u_[0] = clip(kp_ * proportional - kv_ * velocity + ki_ * newIntegral, saturationValue, -saturationValue);

		integral_ = newIntegral;

		return u_;
	}

	const Accumulator<T, MemorySize>& getOutput() const { return u_; }

private:
	T kp_;
	T ki_;
	T kv_;

	Accumulator<T, MemorySize> u_;

	T integral_{0};

	static constexpr T saturationValue = 6.0;
};
