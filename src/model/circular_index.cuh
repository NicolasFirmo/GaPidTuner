#pragma once

template<unsigned MemorySize>
class CircularIndex {
public:
	__device__ constexpr void operator++() { beginIndex_ = beginIndex_-- > 0 ? beginIndex_ : MemorySize - 1; }

	__device__ constexpr unsigned operator[](unsigned index) const { return (beginIndex_ + index) % MemorySize; }

private:
	unsigned beginIndex_ = 0;
};
