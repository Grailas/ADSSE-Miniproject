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

bool verifySort(float arrayToCheck[], unsigned int size)
{
	for (int i = 0; i < size - 1; i++)
	{
		if (arrayToCheck[i] > arrayToCheck[i + 1])
		{
			return false;
		}
	}
	return true;
}

void main()
{
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::uniform_real_distribution<float> distr(-100, 100);

	bool printArrays = false;
	int values = 20000;
	float* sequence = new float[values];

	printf("Generating float sequence of length %d... ", values);

	for (int i = 0; i < values; i++)
	{
		sequence[i] = distr(eng);
	}
	printf("Done.\n\n");

	float* toSort_serial = new float[values];
	std::copy(sequence, sequence + values, toSort_serial);

	float* toSort_parallel = new float[values];
	std::copy(sequence, sequence + values, toSort_parallel);

	if (printArrays) {
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
	}

	printf("Running serial merge sort... ");

	auto start = std::chrono::steady_clock::now();
	MergeSort(toSort_serial, values);
	auto stop = std::chrono::steady_clock::now();
	double elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

	printf("Done.\n");

	printf("Elapsed time:\t%f\n", elapsedTime);

	printf("Checking array sort... ");
	if (verifySort(toSort_serial, values))
	{
		printf("Sorted!\n\n");
	}
	else
	{
		printf("NOT Sorted!\n\n");
	}

	if (printArrays) {
		printf("Result for serial:\n");
		for (int i = 0; i < values; i++)
		{
			printf("%f\t", toSort_serial[i]);
			if ((i + 1) % 5 == 0)
			{
				printf("\n");
			}
		}
		printf("\n\n");
	}


	//Allocate memory for result
	float* result = new float[values];

	printf("Running parallel merge sort...");
	start = std::chrono::steady_clock::now();
	float elapsedKernelTime = RunParallelMergeSort(toSort_parallel, values, result);
	stop = std::chrono::steady_clock::now();
	elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
	printf("Done.\n");

	printf("Elapsed method time:\t%f\n", elapsedTime);
	printf("Elapsed kernel time:\t%f\n", elapsedKernelTime);

	printf("Checking array sort... ");
	if (verifySort(result, values))
	{
		printf("Sorted!\n\n");
	}
	else
	{
		printf("NOT Sorted!\n\n");
	}

	if (printArrays) {
		printf("Result for parallel:\n");
		for (int i = 0; i < values; i++)
		{
			printf("%f\t", result[i]);
			if ((i + 1) % 5 == 0)
			{
				printf("\n");
			}
		}
		printf("\n\n");
	}

	delete[] sequence;
	delete[] toSort_serial;
	delete[] toSort_parallel;
	delete[] result;
}

