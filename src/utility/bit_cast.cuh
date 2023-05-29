#pragma once

#include <device_launch_parameters.h>

#include <type_traits>

template<
	class To, class From,
	typename = std::enable_if_t<(sizeof(To) == sizeof(From)) && (alignof(To) == alignof(From))
								&& std::is_trivially_copyable<From>::value && std::is_trivially_copyable<To>::value>>
__host__ __device__ To bitCast(From& src) noexcept
{
	To	 tgt;
	From staged = src;
	memcpy(&tgt, &staged, sizeof(To));
	return tgt;
}
