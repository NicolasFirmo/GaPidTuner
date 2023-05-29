#include "ga/core.h"
#include "ga/ga.h"
#include "model/model.h"

static constexpr Genome::transform_t transformGene(Genome::dna_t gene, const Genome::transform_t min = Genome::minValue,
												   const Genome::transform_t max = Genome::maxValue)
{
	return min + (max - min) * (Genome::transform_t(gene) / Genome::transform_t(Genome::dnaMax));
}

template<typename T>
constexpr T ramp(const T time, const std::decay_t<T> initialTime, const std::decay_t<T> initialValue,
				 const std::decay_t<T> slope)
{
	return (time - initialTime) * slope + initialValue;
}

Genome::transform_t reference(Genome::transform_t time)
{
	if (time < Genome::transform_t{0.5})
		return 2.0;
	if (time < Genome::transform_t{1.0})
		return 3.0;
	if (time < Genome::transform_t{1.5})
		return 1.0;
	if (time < Genome::transform_t{2.0})
		return 4.5;
	if (time < Genome::transform_t{2.5})
		return 0.5;
	if (time < Genome::transform_t{3.0})
		return ramp(time, 2.0, 3.0, (1.0 / 1.0) * -2.0);
	if (time < Genome::transform_t{3.75})
		return ramp(time, 3.0, 1.0, (1.0 / 0.75) * 3.0);
	if (time < Genome::transform_t{4.5})
		return ramp(time, 3.75, 4.0, (1.0 / 0.75) * -3.5);
	if (time < Genome::transform_t{5.0})
		return ramp(time, 4.5, 0.5, (1.0 / 0.5) * 4.5);

	return 0.0;
}

Genome::fitness_t GA::fitnessFunction(const Genome& genome)
{
	Genome::transform_t kp = transformGene(genome.genes[0]);
	Genome::transform_t ki = transformGene(genome.genes[1]);
	Genome::transform_t kv = transformGene(genome.genes[2]);

	constexpr Genome::transform_t a1 = 0.0;
	constexpr Genome::transform_t a2 = 0.5;
	constexpr Genome::transform_t a3 = 50.0;
	constexpr Genome::transform_t a4 = 1.0;

	Genome::transform_t e1	 = 0.0;
	Genome::transform_t e2	 = 0.0;
	Genome::transform_t iae	 = 0.0;
	Genome::transform_t uDer = 0.0;
	{
		Model model{kp, ki, kv};

		Genome::transform_t vOld = 0.0;
		for (Genome::transform_t t = 0; t < Model::simulationTime; t += Model::ts)
		{
			model.nextStep(reference(t));

			const auto v = model.getControlSignal();

			uDer += (v - vOld) * (v - vOld);
			e1 += v;
			vOld = v;
		}

		e1 /= Model::numberOfSamples;
		uDer /= Model::numberOfSamples;
	}
	{
		Model model{kp, ki, kv};

		for (Genome::transform_t t = 0; t < Model::simulationTime; t += Model::ts)
		{
			model.nextStep(reference(t));

			const auto v = model.getControlSignal();
			const auto e = model.getError();
			e2 += (v - e1) * (v - e1);
			iae += std::abs(e);
		}

		e2 /= Model::numberOfSamples;
		iae /= Model::numberOfSamples;
	}

	// if (tId == 0) {
	//	printf("e1 = %f\n", e1);
	//	printf("e2 = %f\n", e2);
	//	printf("iae = %f\n", iae);
	// }

	return (a1 * e1) + (a2 * e2) + (a3 * iae) + (a4 * uDer);
}

void GA::logGeneration(const size_t genIdx, const GA::population_t& population, const Genome::fitness_t totalFitness,
					   const size_t meanTime)
{
	const auto	meanFitness = totalFitness / static_cast<Genome::fitness_t>(population.size());
	const auto& bestTuning	= *std::max_element(population.begin(), population.end());

	std::cout << '\n';
	std::cout << "generation:  " << genIdx << '\n';
	std::cout << "kp:          " << transformGene(bestTuning.genes[0]) << '\n';
	std::cout << "ki:          " << transformGene(bestTuning.genes[1]) << '\n';
	std::cout << "kv:          " << transformGene(bestTuning.genes[2]) << '\n';
	std::cout << "IAE:         " << 1.0 / bestTuning.fitness << '\n';
	std::cout << "IAE medio:   " << 1.0 / meanFitness << '\n';
	std::cout << "tempo medio: " << meanTime << '\n';
}

int main()
{
	std::setlocale(LC_ALL, ""); // permitir caracteres acentuados na execu��o

	GA ga;

	auto bestTuning = ga.run(500);

	std::cin.ignore();
}
