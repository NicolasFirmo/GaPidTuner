#pragma once

#include <limits>

template<typename T>
static constexpr size_t numberOfBitsIn = sizeof(T) * CHAR_BIT;
