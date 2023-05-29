#pragma once

#include "utility/accumulator.hpp"

template<typename T>
class Plant {
public:
	static constexpr size_t memorySize = 4;

	void update(const Accumulator<T, memorySize>& controlSignal, T ts)
	{
		auto& u = controlSignal;

		y_[0] = (k * ts * ts * u[0] + 2 * k * ts * ts * u[1] + k * ts * ts * u[2] + 8 * tau * y_[1]
				 - (4 * tau - 2 * ts) * y_[2])
			  / (4 * tau + 2 * ts);

		++y_;

		y_[0] = (k * ts * ts * u[0] + 2 * k * ts * ts * u[1] + k * ts * ts * u[2] + 8 * tau * y_[1]
				 - (4 * tau - 2 * ts) * y_[2])
			  / (4 * tau + 2 * ts);
	}

	const Accumulator<T, memorySize>& getOutput() const { return y_; }

private:
	static constexpr T k   = 1.53;
	static constexpr T tau = 0.0414;

	Accumulator<T, memorySize> y_;
};
