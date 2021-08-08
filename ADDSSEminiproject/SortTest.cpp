#include "SerialMergeSort.h"
#include <stdio.h>
#include <random>
#include <chrono>

void main()
{
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::uniform_real_distribution<float> distr(-100, 100);

	int values = 250;
	float* sequence = new float[values];

	printf("Generating float sequence of length %d...\n", values);

	for (int i = 0; i < values; i++)
	{
		sequence[i] = distr(eng);
	}

	float* toSort = new float[values];
	std::copy(sequence, sequence + values, toSort);

	printf("Array to sort:\n");
	for (int i = 0; i < values; i++)
	{
		printf("%f\t", toSort[i]);
		if ((i + 1) % 5 == 0)
		{
			printf("\n");
		}
	}
	printf("\n\n");

	auto start = std::chrono::steady_clock::now();
	MergeSort(toSort, values);
	auto stop = std::chrono::steady_clock::now();
	double elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

	printf("Elapsed time:\t%f\n\n", elapsedTime);

		printf("Result:\n");
	for (int i = 0; i < values; i++)
	{
		printf("%f\t", toSort[i]);
		if ((i + 1) % 5 == 0)
		{
			printf("\n");
		}
	}
	printf("\n");

	delete[] sequence;
	delete[] toSort;
}