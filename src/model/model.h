#pragma once

#include "ga/core.h"
#include "piv.hpp"
#include "plant.hpp"

class Model {
public:
	static constexpr Genome::transform_t sampleRate		 = 60.0;
	static constexpr Genome::transform_t samplePeriod	 = 1.0 / sampleRate;
	static constexpr Genome::transform_t ts				 = samplePeriod / 10.0;
	static constexpr Genome::transform_t simulationTime	 = 5;
	static constexpr size_t				 numberOfSamples = simulationTime / ts;

	Model(Genome::transform_t kp, Genome::transform_t ki, Genome::transform_t kv);

	void nextStep(Genome::transform_t reference);

	[[nodiscard]] Genome::transform_t				   getControlSignal() const { return piv_.getOutput()[0]; }
	[[nodiscard]] Genome::transform_t				   getError() const { return error_[0]; }
	[[nodiscard]] static constexpr Genome::transform_t getSimulationTime() { return simulationTime; }
	[[nodiscard]] static constexpr Genome::transform_t getTimeStep() { return ts; }

private:
	Plant<Genome::transform_t>									   plant_;
	PIV<Genome::transform_t, decltype(plant_)::memorySize>		   piv_;
	Accumulator<Genome::transform_t, decltype(plant_)::memorySize> error_;
};
