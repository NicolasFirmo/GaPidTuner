#include "fitness_function.cuh"

#include "model/model.cuh"

#include <type_traits>

__device__ static constexpr transform_t transformGene(Genome::dna_t gene,
													  const transform_t min = domainMin,
													  const transform_t max = domainMax) {
	return min + (max - min) * (transform_t(gene) / transform_t(Genome::dnaMax));
}

template <typename T>
__device__ constexpr T ramp(const T time, const std::decay_t<T> initialTime,
							const std::decay_t<T> initialValue, const std::decay_t<T> slope) {
	return (time - initialTime) * slope + initialValue;
}
__device__ static constexpr transform_t reference(transform_t time) {
	if (time < transform_t{0.5})
		return 2.0;
	else if (time < transform_t{1.0})
		return 3.0;
	else if (time < transform_t{1.5})
		return 1.0;
	else if (time < transform_t{2.0})
		return 4.5;
	else if (time < transform_t{2.5})
		return 0.5;
	else if (time < transform_t{3.0})
		return ramp(time, 2.0, 3.0, (1.0 / 1.0) * -2.0);
	else if (time < transform_t{3.75})
		return ramp(time, 3.0, 1.0, (1.0 / 0.75) * 3.0);
	else if (time < transform_t{4.5})
		return ramp(time, 3.75, 4.0, (1.0 / 0.75) * -3.5);
	else if (time < transform_t{5.0})
		return ramp(time, 4.5, 0.5, (1.0 / 0.5) * 4.5);
}

__device__ Genome::fitness_t fitnessFunction(const Genome &genome, unsigned tId) {
	transform_t kp = transformGene(genome.genes[0]);
	transform_t ki = transformGene(genome.genes[1]);
	transform_t kv = transformGene(genome.genes[2]);

	constexpr transform_t a1 = 0.0;
	constexpr transform_t a2 = 0.5;
	constexpr transform_t a3 = 50.0;
	constexpr transform_t a4 = 1.0;

	transform_t e1 = 0.0;
	transform_t e2 = 0.0;
	transform_t iae = 0.0;
	transform_t uDer = 0.0;
	{
		Model model{kp, ki, kv};
		transform_t vOld = 0.0;
		for (transform_t t = 0; t < model.simulationTime; t += model.ts) {
			model.nextStep(reference(t));
			const auto v = model.measureControlSignal();
			uDer += (v - vOld) * (v - vOld);
			e1 += v;
			vOld = v;
		}
		e1 /= model.numberOfSamples;
		uDer /= model.numberOfSamples;
	}
	{
		Model model{kp, ki, kv};
		for (transform_t t = 0; t < model.simulationTime; t += model.ts) {
			model.nextStep(reference(t));
			const auto v = model.measureControlSignal();
			const auto e = model.measureError();
			e2 += (v - e1) * (v - e1);
			iae += std::abs(e);
		}
		e2 /= model.numberOfSamples;
		iae /= model.numberOfSamples;
	}

	// if (tId == 0) {
	//	printf("e1 = %f\n", e1);
	//	printf("e2 = %f\n", e2);
	//	printf("iae = %f\n", iae);
	// }

	return a1 * e1 + a2 * e2 + a3 * iae + a4 * uDer;
}