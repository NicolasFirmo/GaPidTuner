#pragma once
#include <limits>

#include <cstdint>

struct Genome {
	using dna_t = uint32_t;
	using fitness_t = float;
	static constexpr unsigned numberOfGenes = 3;
	static constexpr auto dnaMax = std::numeric_limits<Genome::dna_t>::max();

	dna_t genes[numberOfGenes]; // do not change
	fitness_t fitness;			// do not change

	constexpr bool operator<(const Genome& rhs) const { return fitness < rhs.fitness; }
	constexpr bool operator>(const Genome& rhs) const { return fitness > rhs.fitness; }
};

using transform_t = float;

static constexpr transform_t domainMin = 0.0;
static constexpr transform_t domainMax = 100.0;