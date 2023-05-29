#pragma once

#include <compare>
#include <limits>

struct Genome {
	using dna_t		  = uint64_t;
	using fitness_t	  = float;
	using transform_t = float;

	static constexpr size_t		 numberOfGenes = 3;
	static constexpr transform_t minValue	   = 0.0;
	static constexpr transform_t maxValue	   = 100.0;

	static constexpr auto dnaMax = std::numeric_limits<Genome::dna_t>::max();

	std::array<dna_t, numberOfGenes> genes{};
	fitness_t						 fitness{0};

	bool operator==(const Genome& rhs) const { return fitness == rhs.fitness; }
	auto operator<=>(const Genome& rhs) const { return fitness <=> rhs.fitness; }
};
