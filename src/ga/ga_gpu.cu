#include "ga_gpu.h"

#include "fitness_function.cuh"

#include "utility/cuda_core.cuh"
#include "utility/timer.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <execution>

namespace {

// Seeds the states. via
// https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example
__global__ void initStates(const unsigned populationSize, curandState* state)
{
	const auto tId = blockIdx.x * blockDim.x + threadIdx.x;

	/* Each thread gets same seed, a different sequence
	   number, no offset */
	if (tId < populationSize)
		curand_init(1234, tId, 0, &state[tId]); // NOLINT: 1234 is a very magical number indeed!
}

__device__ Genome generateGenome(curandState* state)
{
	Genome genome{};

	for (auto& gene : genome.genes)
		gene = curand(state);

	return genome;
}

__global__ void populate(Genome* population, const unsigned populationSize, curandState* state)
{
	const auto tId = blockIdx.x * blockDim.x + threadIdx.x;

	if (tId < populationSize)
	{
		curandState* localState = &state[tId];

		population[tId] = generateGenome(localState);

		state[tId] = *localState;
	}
}

__global__ void calculateFitneess(Genome* population, const unsigned populationSize, const unsigned eliteSize = 0)
{
	const auto tId = blockIdx.x * blockDim.x + threadIdx.x;

	if (tId < populationSize - eliteSize)
	{
		auto genome = population[tId];

		genome.fitness = fitnessFunction(genome, tId);

		population[tId] = genome;
	}
}

template<typename T>
__device__ void warpSumReduce(volatile T* sharedData, unsigned tId)
{
	sharedData[tId] += sharedData[tId + 32];
	sharedData[tId] += sharedData[tId + 16];
	sharedData[tId] += sharedData[tId + 8];
	sharedData[tId] += sharedData[tId + 4];
	sharedData[tId] += sharedData[tId + 2];
	sharedData[tId] += sharedData[tId + 1];
}
template<typename T>
__device__ void sumReduce(volatile T* sharedData, unsigned blocDim, unsigned tId)
{
	for (auto s = blocDim / 2; s > GAGPU::warpSize; s >>= 1U)
	{
		if (tId < s)
			sharedData[tId] += sharedData[tId + s];
		__syncthreads();
	}

	if (tId < GAGPU::warpSize)
		warpSumReduce(sharedData, tId);
}

__global__ void fitnessReduceStep1(Genome* population, const unsigned populationSize,
								   Genome::fitness_t* fitnessCumulative)
{
	__shared__ Genome::fitness_t sharedSum[GAGPU::blockSize];

	const auto tId = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	if (tId < populationSize)
		sharedSum[threadIdx.x] = population[tId].fitness;
	else
		sharedSum[threadIdx.x] = 0;

	if (tId + blockDim.x < populationSize)
		sharedSum[threadIdx.x] += population[tId + blockDim.x].fitness;

	__syncthreads();

	sumReduce(sharedSum, blockDim.x, threadIdx.x);

	if (threadIdx.x == 0)
		fitnessCumulative[blockIdx.x] = sharedSum[0];
}

__global__ void fitnessReduceStep2(Genome::fitness_t* fitnessCumulative, const unsigned fitnessCumulativeSize)
{
	__shared__ Genome::fitness_t sharedSum[GAGPU::blockSize];

	const auto tId = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	if (tId < fitnessCumulativeSize)
		sharedSum[threadIdx.x] = fitnessCumulative[tId];
	else
		sharedSum[threadIdx.x] = 0;

	if (tId + blockDim.x < fitnessCumulativeSize)
		sharedSum[threadIdx.x] += fitnessCumulative[tId + blockDim.x];

	__syncthreads();

	sumReduce(sharedSum, blockDim.x, threadIdx.x);

	if (threadIdx.x == 0)
		fitnessCumulative[blockIdx.x] = sharedSum[0];
}

__device__ Genome rouletteSelect(Genome* population, const Genome::fitness_t cumulative, curandState* state)
{
	const Genome::fitness_t selectionLocation = curand_uniform(state) * cumulative;
	Genome::fitness_t		selectionIndex	  = 0;

	for (unsigned i = 0;; i++)
	{
		selectionIndex += population[i].fitness;
		if (selectionIndex >= selectionLocation)
			return population[i];
	}
}

__device__ void crossOver(Genome& genomeA, Genome& genomeB, const unsigned crossPoint)
{
	const Genome::gene_t tailMask = ~(Genome::geneMax << crossPoint);

	for (unsigned i = 0; i < Genome::numberOfGenes; i++)
	{
		const Genome::gene_t difference	= genomeA.genes[i] ^ genomeB.genes[i];
		const Genome::gene_t geneChanger = difference & tailMask;
		genomeA.genes[i] ^= geneChanger;
		genomeB.genes[i] ^= geneChanger;
	}
}

__device__ void mutate(Genome& genome, const float mutationChance, curandState* state)
{
	for (auto& gene : genome.genes)
		for (unsigned i = 0; i < numberOfBitsIn<Genome::gene_t>; i++)
			if (curand_uniform(state) <= mutationChance)
				gene ^= (Genome::gene_t(1) << i);
}

__global__ void reproduce(Genome* population, const unsigned populationSize, const unsigned eliteSize,
						  const float mutationChance, const Genome::fitness_t* fitnessCumulative,
						  Genome::fitness_t greatestFitness, curandState* state)
{
	__shared__ Genome sharedParents[GAGPU::blockSize];

	const auto tId = blockIdx.x * blockDim.x + threadIdx.x;

	const auto totalFitness = Genome::fitness_t(populationSize) - (fitnessCumulative[0] / greatestFitness);

	if (tId == populationSize - 1)
	{
		printf("Mean fitness: %.8f\n", fitnessCumulative[0] / Genome::fitness_t(populationSize));
		printf("Best fitness: %.8f\n", population[populationSize - 1].fitness);
		printf("Wrost fitness: %.8f\n", greatestFitness);
	}

	Genome::fitness_t fitness;
	if (tId < populationSize)
	{
		fitness					= population[tId].fitness;
		population[tId].fitness = 1.0F - (fitness / greatestFitness);
	}

	if (tId < populationSize - eliteSize)
	{
		curandState* localState = &state[tId];

		sharedParents[threadIdx.x] = rouletteSelect(population, totalFitness, localState);

		__syncthreads();

		if (threadIdx.x % 2 == 0)
		{
			const auto crossOverMidPoint = static_cast<unsigned>(
				curand_uniform(localState) * (numberOfBitsIn<Genome::gene_t> - 2) + 1);

			crossOver(sharedParents[threadIdx.x], sharedParents[threadIdx.x + 1], crossOverMidPoint);
		}

		__syncthreads();

		mutate(sharedParents[threadIdx.x], mutationChance, localState);

		population[tId] = sharedParents[threadIdx.x];
		state[tId]		= *localState;
	}

	if (tId < populationSize && tId >= populationSize - eliteSize)
		population[tId].fitness = fitness;
}

} // namespace

GAGPU::GAGPU(const unsigned populationSize, const unsigned eliteSize, const float mutationChancePerGene)
	: populationSize_(populationSize),
	  eliteSize_(eliteSize),
	  mutationChance_(mutationChancePerGene / numberOfBitsIn<decltype(Genome::genes)>),
	  populationHost_(populationSize)
{
	cudaCall(cudaMalloc(&populationDev_, sizeof(Genome) * populationSize_));
	cudaCall(cudaMalloc(&fitnessCumulative_, sizeof(Genome::fitness_t) * fitnessCumulativeSize_));
	cudaCall(cudaMalloc(&stateDev_, sizeof(curandState) * populationSize_));

	initStates<<<gridSize_, GAGPU::blockSize>>>(populationSize_, stateDev_);
	afterKernelCall();

	generatePopulation();
}

void GAGPU::generatePopulation()
{
	populate<<<gridSize_, GAGPU::blockSize>>>(populationDev_, populationSize_, stateDev_);
	afterKernelCall();

	calculateFitneess<<<gridSize_, GAGPU::blockSize>>>(populationDev_, populationSize_);
	afterKernelCall();
}

std::vector<Genome> GAGPU::getPopulation()
{
	cudaCall(
		cudaMemcpy(populationHost_.data(), populationDev_, sizeof(Genome) * populationSize_, cudaMemcpyDeviceToHost));

	return populationHost_;
}

GAGPU::~GAGPU()
{
	cudaCall(cudaFree(stateDev_));
	cudaCall(cudaFree(fitnessCumulative_));
	cudaCall(cudaFree(populationDev_));
}

void GAGPU::run(const unsigned numberOfGenerations)
{
	AccTimer totalTime{"generation loop"};

	for (unsigned generation = 0; generation < numberOfGenerations; generation++)
	{
		printf("\ngeneration: %u\n", generation);

		Timer t{"generation loop"};

		const auto greatestFitness = getGreatestFitnessAndSelectElite();

		fitnessReduceStep1<<<fitnessCumulativeSize_, GAGPU::blockSize>>>(populationDev_, populationSize_,
																		 fitnessCumulative_);
		afterKernelCall();

		fitnessReduceStep2<<<1, GAGPU::blockSize>>>(fitnessCumulative_, fitnessCumulativeSize_);
		afterKernelCall();

		// TEST
		//{
		//	cudaDeviceSynchronize();

		//	getPopulation();

		//	auto totalFitness = std::transform_reduce(
		//		populationHost_.begin(), populationHost_.end(),
		// Genome::fitness_t{0}, std::plus{},
		//		[](const Genome &genome) { return genome.fitness; });
		//	printf("total fitness CPU: %.8f\n", totalFitness);

		//}

		reproduce<<<gridSizeNoElite_, GAGPU::blockSize>>>(populationDev_, populationSize_, eliteSize_, mutationChance_,
														  fitnessCumulative_, greatestFitness, stateDev_);
		afterKernelCall();

		cudaDeviceSynchronize();

		calculateFitneess<<<gridSizeNoElite_, GAGPU::blockSize>>>(populationDev_, populationSize_, eliteSize_);
		afterKernelCall();

		totalTime.accumulateCount();
		const auto meanTime = totalTime.getTotalDuration() / (generation + 1);

		printf("\nmean loop time: %zu\n", meanTime);
	}
}

Genome::fitness_t GAGPU::getGreatestFitnessAndSelectElite()
{
	cudaCall(
		cudaMemcpy(populationHost_.data(), populationDev_, sizeof(Genome) * populationSize_, cudaMemcpyDeviceToHost));

	std::partial_sort_copy(std::execution::par_unseq, populationHost_.rbegin(), populationHost_.rend(),
						   populationHost_.rbegin(), populationHost_.rbegin() + eliteSize_);

	cudaCall(
		cudaMemcpy(populationDev_, populationHost_.data(), sizeof(Genome) * populationSize_, cudaMemcpyHostToDevice));

	const Genome& wrost = *std::max_element(std::execution::par_unseq, populationHost_.begin(), populationHost_.end(),
											[](const Genome& a, const Genome& b) { return a.fitness < b.fitness; });

	return wrost.fitness;
}
