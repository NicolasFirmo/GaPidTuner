#pragma once
#include "piv.cuh"
#include "plant.cuh"

#include "ga/ga_core.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class Model {
public:
	__device__ Model(transform_t kp, transform_t ki, transform_t kv);

	__device__ void nextStep(transform_t reference);

	[[nodiscard]] __device__ transform_t measureError() const {
		const auto i0 = sharedIndex_[0];
		return error_[i0];
	}

	[[nodiscard]] __device__ transform_t measureControlSignal() const {
		const auto i0 = sharedIndex_[0];
		return piv_.getControlSignal()[i0];
	}

	static constexpr transform_t fps = 60.0F;
	static constexpr transform_t samplePeriod = 1.0F / fps;
	static constexpr transform_t ts = samplePeriod / 10.0F;
	static constexpr transform_t simulationTime = 5.0F;
	static constexpr unsigned numberOfSamples = unsigned(simulationTime / ts);

private:
	Plant<transform_t> plant_{};
	PIV<transform_t, decltype(plant_)::memorySize> piv_;
	transform_t error_[decltype(plant_)::memorySize]{};
	CircularIndex<decltype(plant_)::memorySize> sharedIndex_;
};