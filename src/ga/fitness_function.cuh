#pragma once
#include "ga_core.h"

#include <device_launch_parameters.h>

__device__ Genome::fitness_t fitnessFunction(const Genome& genome, unsigned tId);