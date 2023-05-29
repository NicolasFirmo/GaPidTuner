#include "model.h"

Model::Model(Genome::transform_t kp, Genome::transform_t ki, Genome::transform_t kv) : piv_(kp, ki, kv) {}

void Model::nextStep(const Genome::transform_t reference)
{
	++error_;

	const auto& measuredY = plant_.getOutput();

	error_[0] = reference - measuredY[0];

	piv_.update(error_, measuredY, ts);

	plant_.update(piv_.getOutput(), ts);
}
