#pragma once

template<typename T>
constexpr T clip(T value, T upperLimit, T lowerLimit)
{
	return value > upperLimit ? upperLimit : value < lowerLimit ? lowerLimit : value;
}