#pragma once

#include <array>

template<typename T, size_t MemorySize>
class Accumulator {
public:
	Accumulator() = default;

	void operator++() { currentIndex_ = currentIndex_-- > 0 ? currentIndex_ : MemorySize - 1; }

	T&		 operator[](size_t index) { return values_[(currentIndex_ + index) % MemorySize]; }
	const T& operator[](size_t index) const { return values_[(currentIndex_ + index) % MemorySize]; }

private:
	size_t currentIndex_ = MemorySize - 1;

	std::array<T, MemorySize> values_{};
};
