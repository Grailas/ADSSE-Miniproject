#include <stdio.h>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "SerialMergeSort.h"


//An error handling macro, for debugging
#define CHECK(call)															\
{																			\
	const cudaError_t error = call;											\
	if (error != cudaSuccess)												\
	{																		\
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
		exit(1);															\
	}																		\
}

//Declaration
extern double RunParallelMergeSort(float* toSort, unsigned int values, float* result);

void main()
{
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::uniform_real_distribution<float> distr(-100, 100);

	int values = 5;
	float* sequence = new float[values];

	printf("Generating float sequence of length %d...\n", values);

	for (int i = 0; i < values; i++)
	{
		sequence[i] = distr(eng);
	}

	float* toSort_serial = new float[values];
	std::copy(sequence, sequence + values, toSort_serial);

	float* toSort_parallel = new float[values];
	std::copy(sequence, sequence + values, toSort_parallel);

	printf("Array to sort:\n");
	for (int i = 0; i < values; i++)
	{
		printf("%f\t", sequence[i]);
		if ((i + 1) % 5 == 0)
		{
			printf("\n");
		}
	}
	printf("\n\n");

	auto start = std::chrono::steady_clock::now();
	MergeSort(toSort_serial, values);
	auto stop = std::chrono::steady_clock::now();
	double elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

	printf("Elapsed time:\t%f\n\n", elapsedTime);

	printf("Result for serial:\n");
	for (int i = 0; i < values; i++)
	{
		printf("%f\t", toSort_serial[i]);
		if ((i + 1) % 5 == 0)
		{
			printf("\n");
		}
	}
	printf("\n");

	//Allocate memory for result
	float* result = new float[values];

	elapsedTime = RunParallelMergeSort(toSort_parallel, values, result);

	printf("Elapsed time:\t%f\n\n", elapsedTime);

	printf("Result for parallel:\n");
	for (int i = 0; i < values; i++)
	{
		printf("%f\t", toSort_parallel[i]);
		if ((i + 1) % 5 == 0)
		{
			printf("\n");
		}
	}
	printf("\n");

	delete[] sequence;
	delete[] toSort_serial;
	delete[] toSort_parallel;
	delete[] result;
}