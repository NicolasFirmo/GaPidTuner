#pragma once

#include "ga_core.h"

#include "utility/type_bits.h"

#include <vector>

struct curandStateXORWOW;
using curandState = struct curandStateXORWOW;

class GAGPU {
public:
	static constexpr unsigned warpSize	= 32U;
	static constexpr unsigned blockSize = 32U * 4;

	explicit GAGPU(unsigned populationSize = 10'000, unsigned eliteSize = 10, float mutationChancePerGene = 0.01F);
	~GAGPU();

	GAGPU(const GAGPU&)			   = delete;
	GAGPU& operator=(const GAGPU&) = delete;
	GAGPU(GAGPU&&)				   = delete;
	GAGPU& operator=(GAGPU&&)	   = delete;

	void run(unsigned numberOfGenerations);

	void generatePopulation();

	[[nodiscard]] std::vector<Genome> getPopulation();

private:
	[[nodiscard]] Genome::fitness_t getGreatestFitnessAndSelectElite();

	unsigned populationSize_;
	unsigned eliteSize_;
	float	 mutationChance_;

	unsigned gridSizeNoElite_ = (populationSize_ - eliteSize_ + (blockSize - 1)) / blockSize;
	unsigned gridSize_		  = (populationSize_ + (blockSize - 1)) / blockSize;

	unsigned fitnessCumulativeSize_ = (gridSize_ + (2 - 1)) / 2;

	Genome*			   populationDev_	  = nullptr;
	Genome::fitness_t* fitnessCumulative_ = nullptr;
	curandState*	   stateDev_		  = nullptr;

	std::vector<Genome> populationHost_;
};
