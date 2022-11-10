#pragma once

template<typename T>
struct ValueRange {
	T lower, upper;
};
template<typename T>
__host__ __device__ constexpr T saturate(T value, ValueRange<T> limit) {
	return limit.upper < value ? limit.upper : value < limit.lower ? limit.lower : value;
}

template<typename Float>
__host__ __device__ constexpr Float smallFloor(const Float x) {
	static_assert(std::is_floating_point_v<Float>,
				  "Template argument Float must be a floating point type!");
	return Float(int(x));
}

template<typename Float>
__host__ __device__ constexpr Float quantize(const Float value, const Float range,
											 const int resolution) {
	static_assert(std::is_floating_point_v<Float>,
				  "Template argument Float must be a floating point type!");
	const Float quantizedValue = smallFloor(value / range * resolution);
	return quantizedValue * range / resolution;
}