#include "ga/ga_gpu.h"

static constexpr Genome::transform_t transformGene(Genome::gene_t gene, const Genome::transform_t min = Genome::minValue,
												   const Genome::transform_t max = Genome::maxValue)
{
	return min + (max - min) * (Genome::transform_t(gene) / Genome::transform_t(Genome::geneMax));
}

static constexpr size_t sampleSize = 10;

template<typename Container>
requires std::is_same_v<typename Container::value_type, Genome>
void writeSampleToFile(const Container& sample, Genome::fitness_t totalIAE, size_t populationSize,
					   std::string_view sampleName, std::ostream& file = std::cout)
{
	file << '\n' << sampleName << "\n\n";

	for (auto& genome : sample)
	{
		file << "kp = " << transformGene(genome.genes[0]) << '\n';
		file << "ki = " << transformGene(genome.genes[1]) << '\n';
		file << "kv = " << transformGene(genome.genes[2]) << '\n';
		file << "IAE: " << genome.fitness << '\n';
		file << "Mean IAE: " << totalIAE / Genome::fitness_t(populationSize) << '\n';
	}
}

int main()
{
	std::setlocale(LC_ALL, ""); // permitir caracteres acentuados na execução

	unsigned populationSize;
	std::cout << "population size: ";
	std::cin >> populationSize;

	unsigned eliteSize;
	std::cout << "elite size: ";
	std::cin >> eliteSize;

	float mutationChancePerGene;
	std::cout << "mutation chance per gene (%): ";
	std::cin >> mutationChancePerGene;

	mutationChancePerGene /= 100.0F;

	GAGPU ga{populationSize, eliteSize, mutationChancePerGene};

	unsigned numberOfGenerations;
	while (true)
	{
		std::cout << "number of generations (0 to terminate): ";
		std::cin >> numberOfGenerations;

		if (numberOfGenerations == 0)
			break;

		ga.run(numberOfGenerations);
	}

	auto population = ga.getPopulation();

	decltype(population) bestSample(sampleSize);
	std::partial_sort_copy(std::execution::par_unseq, population.begin(), population.end(), bestSample.begin(),
						   bestSample.end());

	decltype(population) worstSample(sampleSize);
	std::partial_sort_copy(std::execution::par_unseq, population.begin(), population.end(), worstSample.begin(),
						   worstSample.end(), std::greater{});

	const auto totalIAE = std::transform_reduce(std::execution::par_unseq, population.begin(), population.end(),
												Genome::fitness_t{0}, std::plus{},
												[](const Genome& genome) { return genome.fitness; });

	std::ofstream populationRank{"population_rank.txt", std::ios_base::trunc};

	writeSampleToFile(bestSample, totalIAE, population.size(), "Best", populationRank);
	writeSampleToFile(worstSample, totalIAE, population.size(), "Worst", populationRank);
}
