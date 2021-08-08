#pragma once

__device__ unsigned int BinarySearch(float value, float* array, unsigned int subarrayLow, unsigned int subarrayHigh);

__global__ void ParallelMerge(float* toSort, unsigned int leftLow, unsigned int leftHigh, unsigned int rightLow, unsigned int rightHigh, float* result, unsigned int resultLow);

__global__ void ParallelMergeSort(float* toSort, unsigned int lowIndex, unsigned int highIndex, float* result, unsigned int resultLowIndex);