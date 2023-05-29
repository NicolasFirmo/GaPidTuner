#pragma once

#include "core.h"

#include "utility/random.hpp"
#include "utility/typebits.h"

class GA {
public:
	using population_t = std::vector<Genome>;

	static constexpr size_t mutationChanceOfOneIn = numberOfBitsIn<decltype(Genome::genes)> * 50;
	static constexpr size_t populationSize		  = 10000;
	static constexpr size_t eliteSize			  = 50;

	Genome run(size_t numberOfGenerations);

	static Genome::fitness_t fitnessFunction(const Genome& genome);

	static void logGeneration(const size_t genIdx, const GA::population_t& population,
							  const Genome::fitness_t totalFitness, const size_t meanTime);

private:
	void populate();

	Genome::fitness_t getTotalFitness() const;

	Genome rouletteSelect(Genome::fitness_t cumulativeFitness) const;

	static void crossOver(Genome& genomeA, Genome& genomeB, size_t crossPoint);

	void mutate(Genome& genome) const;

	population_t   population_{};
	mutable Random generator_;
};
