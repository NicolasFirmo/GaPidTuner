#include "ga.h"

#include "utility/parallel.hpp"
#include "utility/timer.h"

#include <cassert>

Genome GA::run(const size_t numberOfGenerations)
{
	populate();

	population_t newPopulation(populationSize);
	population_t elite(eliteSize);

	AccTimer totalTime{"generation loop"};

	for (size_t generation = 0; generation < numberOfGenerations; generation++)
	{
		Timer t{"generation loop"};

		par::parallelProcess(
			[&](par::thread_id_t tId)
			{
				for (size_t i = tId; i < populationSize; i += par::nThreads)
					population_[i].fitness = fitnessFunction(population_[i]);
			});

		std::partial_sort_copy(std::execution::par_unseq, population_.begin(), population_.end(), elite.begin(),
							   elite.end(), std::greater{});

		const auto	 totalFitness	   = getTotalFitness();
		const size_t crossOverMidPoint = generator_.generate(1UI64, numberOfBitsIn<Genome::dna_t> - 1UI64);

		par::parallelProcess(
			[&](par::thread_id_t tId)
			{
				for (size_t i = tId; i < (populationSize - eliteSize) / 2; i += par::nThreads)
				{
					auto parentA = rouletteSelect(totalFitness);
					auto parentB = rouletteSelect(totalFitness);
					crossOver(parentA, parentB, crossOverMidPoint);
					mutate(parentA);
					mutate(parentB);
					newPopulation[2 * i]	 = parentA;
					newPopulation[2 * i + 1] = parentB;
				}
			});

		std::copy(std::execution::par_unseq, elite.rbegin(), elite.rend(), newPopulation.rbegin());

		totalTime.accumulate();
		const auto meanTime = totalTime.getTotalDuration() / (generation + 1);

		logGeneration(generation, newPopulation, totalFitness, meanTime);

		std::swap(population_, newPopulation);
	}

	return *std::max_element(std::execution::par_unseq, population_.begin(), population_.end());
}

void GA::populate()
{
	population_.resize(populationSize);
	for (auto& genome : population_)
		for (auto& gene : genome.genes)
			gene = generator_.generate(std::numeric_limits<Genome::dna_t>::max());
}

Genome::fitness_t GA::getTotalFitness() const
{
	return std::transform_reduce(std::execution::par_unseq, population_.begin(), population_.end(), .0, std::plus{},
								 [](const Genome& value) { return value.fitness; });
}

Genome GA::rouletteSelect(Genome::fitness_t cumulativeFitness) const
{
	const Genome::fitness_t selectionLocation = generator_.generate(cumulativeFitness);
	Genome::fitness_t		selectionIndex	  = 0;

	for (auto&& genome : population_)
	{
		selectionIndex += genome.fitness;
		if (selectionIndex > selectionLocation)
			return genome;
	}

	assert(false && "This code shouldn't be reached!");
}

void GA::crossOver(Genome& genomeA, Genome& genomeB, const size_t crossPoint)
{
	const Genome::dna_t tailMask = ~(std::numeric_limits<Genome::dna_t>::max() << crossPoint);

	for (size_t i = 0; i < Genome::numberOfGenes; i++)
	{
		const Genome::dna_t difference	= genomeA.genes[i] ^ genomeB.genes[i];
		const Genome::dna_t geneChanger = difference & tailMask;
		genomeA.genes[i] ^= geneChanger;
		genomeB.genes[i] ^= geneChanger;
	}
}

void GA::mutate(Genome& genome) const
{
	for (auto& gene : genome.genes)
		for (size_t i = 0; i < numberOfBitsIn<Genome::dna_t>; i++)
			if (generator_.chanceOfOneIn(mutationChanceOfOneIn))
				gene ^= (Genome::dna_t(1) << i);
}
