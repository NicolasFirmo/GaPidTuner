#pragma once

#include <random>
#include <type_traits>

template<typename T>
concept Real = std::is_floating_point_v<T>;

template<typename T>
concept Int = std::is_integral_v<T>;

template<typename T>
concept Uint = std::is_unsigned_v<T>;

class Random {
public:
	template<Real RealT>
	RealT generate(const RealT max)
	{
		return std::uniform_real_distribution<RealT>{0, max}(generator_);
	}

	template<Uint UintT>
	UintT generate(const UintT max)
	{
		return std::uniform_int_distribution<UintT>{0, max}(generator_);
	}

	template<Real RealT>
	RealT generate(const RealT min, const RealT max)
	{
		return std::uniform_real_distribution<RealT>{min, max}(generator_);
	}

	template<Int IntT>
	IntT generate(const IntT min, const IntT max)
	{
		return std::uniform_int_distribution<IntT>{min, max}(generator_);
	}

	template<Uint UintT>
	bool chanceOfOneIn(const UintT amount)
	{
		return std::uniform_int_distribution<UintT>{0, amount - 1}(generator_) < 1;
	}

private:
	std::mt19937_64 generator_{std::random_device{}()};
};
