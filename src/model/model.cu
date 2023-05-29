#include "model.cuh"

__device__ Model::Model(Genome::transform_t kp, Genome::transform_t ki, Genome::transform_t kv) : piv_(kp, ki, kv) {}

__device__ void Model::nextStep(const Genome::transform_t reference)
{
	++sharedIndex_;

	const unsigned indexArray[decltype(plant_)::memorySize] = {sharedIndex_[0], sharedIndex_[1], sharedIndex_[2],
															   sharedIndex_[3]};

	auto measuredY = plant_.getOutput();

	error_[indexArray[0]] = reference - measuredY[indexArray[0]];

	piv_.update(error_, measuredY, ts, indexArray);

	plant_.update(piv_.getControlSignal(), ts, indexArray);
}
