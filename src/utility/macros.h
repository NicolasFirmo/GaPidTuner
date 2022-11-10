#pragma once

#if defined(_MSC_VER)
	#define debugBreak() __debugbreak()
	#define funcSign	 __FUNCSIG__
#elif defined(__GNUC__) || defined(__clang__)
	#define debugBreak() __builtin_trap()
	#define funcSign	 __PRETTY_FUNCTION__
#else
	#error Only MSVC, GCC and Clang supported for now!
#endif
