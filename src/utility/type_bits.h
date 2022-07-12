#pragma once
#include <limits>

template <typename T>
static constexpr unsigned numberOfBitsIn = sizeof(T) * CHAR_BIT;