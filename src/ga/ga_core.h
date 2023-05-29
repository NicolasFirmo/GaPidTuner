#pragma once

#include <cstdint>
#include <limits>

struct Genome {
	using gene_t	  = uint32_t;
	using fitness_t	  = float;
	using transform_t = float;

	static constexpr unsigned	 numberOfGenes = 3;
	static constexpr transform_t minValue	   = 0.0;
	static constexpr transform_t maxValue	   = 100.0;

	static constexpr auto geneMax = std::numeric_limits<gene_t>::max();

	gene_t	  genes[numberOfGenes];
	fitness_t fitness;

	constexpr bool operator<(const Genome& rhs) const { return fitness < rhs.fitness; }
	constexpr bool operator>(const Genome& rhs) const { return fitness > rhs.fitness; }
};
