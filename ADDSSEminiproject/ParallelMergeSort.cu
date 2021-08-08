#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <thrust/swap.h>
#include "ParallelMergeSort.cuh"



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

//An error handling macro, for debugging
#define CHECK_NO_EXIT(call)													\
{																			\
	const cudaError_t error = call;											\
	if (error != cudaSuccess)												\
	{																		\
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
	}																		\
}

__device__ unsigned int BinarySearch(float value, float* array, unsigned int subarrayLow, unsigned int subarrayHigh)
{
	unsigned int low = subarrayLow, high = umax(subarrayLow, subarrayHigh + 1);
	while (low < high)
	{
		unsigned int mid = (low + high) / 2;
		if (value <= array[mid])
		{
			high = mid;
		}
		else
		{
			low = mid + 1;
		}
	}
	return high;
}

/// <summary>
/// Parallel merge operation based on Introduction to Algorithms, Third Edition, section 27.3
/// </summary>
/// <param name="toSort">T: Array of data to sort.</param>
/// <param name="leftLow">p1: Lowest index of left subarray.</param>
/// <param name="leftHigh">r1: Highest index of left subarray.</param>
/// <param name="rightLow">p2: Lowest index of right subarray.</param>
/// <param name="rightHight">r2: Highest index of right subarray.</param>
/// <param name="result">A: Subarray to store result.</param>
/// <param name="resultLow">p3: Lowest index of result subarray.</param>
/// <returns></returns>
__global__ void ParallelMerge(float* toSort, unsigned int leftLow, unsigned int leftHigh, unsigned int rightLow, unsigned int rightHigh, float* result, unsigned int resultLow)
{
	unsigned int leftSize = leftHigh - leftLow + 1,
		rightSize = rightHigh - rightLow + 1;

	if (leftSize < rightSize)	//Ensure that left subarray is >= than right subarray
	{
		thrust::swap(leftLow, rightLow);
		thrust::swap(leftHigh, rightHigh);
		thrust::swap(leftSize, rightSize);
	}
	if (leftSize == 0)	//Are both arrays empty?
	{
		return;
	}
	else
	{
		unsigned int leftQ = (leftLow + leftHigh) / 2; //midpoint of left subarray
		unsigned int rightQ = BinarySearch(toSort[leftQ], toSort, rightLow, rightHigh);
		unsigned int resultQ = resultLow + (leftQ - leftLow) + (rightQ - rightLow);
		result[resultQ] = toSort[leftQ];
		ParallelMerge << <1, 1 >> > (toSort, leftLow, leftQ - 1, rightLow, rightQ - 1, result, resultLow);
		ParallelMerge << <1, 1 >> > (toSort, leftQ + 1, leftHigh, rightQ, rightHigh, result, resultQ + 1);
		//Implicit sync
	}
}

__global__ __device__ void ParallelMergeSort(float* toSort, unsigned int lowIndex, unsigned int highIndex, float* result, unsigned int resultLowIndex)
{
	unsigned int n = highIndex - lowIndex + 1;
	if (n == 1)
	{
		result[resultLowIndex] = toSort[lowIndex];
	}
	else
	{
		float* subarray;
		CHECK_NO_EXIT(cudaMalloc(&subarray, n * sizeof(float)));
		unsigned int qDivide = (lowIndex + highIndex) / 2;
		unsigned int subarrayQ = qDivide - lowIndex + 1;
		ParallelMergeSort << <1, 1 >> > (toSort, lowIndex, qDivide, subarray, 1);
		ParallelMergeSort << <1, 1 >> > (toSort, qDivide + 1, highIndex, subarray, subarrayQ + 1);
		CHECK_NO_EXIT(cudaDeviceSynchronize());
		ParallelMerge << <1, 1 >> > (subarray,
			1,
			subarrayQ,
			subarrayQ + 1,
			n,
			result,
			resultLowIndex);
		CHECK_NO_EXIT(cudaDeviceSynchronize());
		cudaFree(&subarray);
	}
}

extern void test()
{
	printf("Hello");
}

extern double RunParallelMergeSort(float* toSort, unsigned int values, float* result)
{
	printf("Running parallel merge sort...\n\n");
	//Allocate memory for device
	float* d_toSort;
	CHECK(cudaMalloc(&d_toSort, values * sizeof(float)));
	float* d_result;
	CHECK(cudaMalloc(&d_result, values * sizeof(float)));

	
	//cudaDeviceLimit(cudaLimitDevRuntimeSyncDepth, 4);

	auto start = std::chrono::steady_clock::now();
	ParallelMergeSort << <1, 1 >> > (d_toSort, 0, values - 1, d_result, 0);
	auto stop = std::chrono::steady_clock::now();
	double elapsedTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

	cudaMemcpy(result, d_result, values * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_toSort);
	cudaFree(d_result);

	return elapsedTime;
}