#include <malloc.h>
#include <algorithm>
#include "SerialMergeSort.h"

void merge(float toSort[], unsigned int leftIndex, unsigned int centerIndex, unsigned int rightIndex)
{
	unsigned int leftSplitSize = centerIndex - leftIndex + 1,
		rightSplitSize = rightIndex - centerIndex;

	float* leftSplit = (float*)malloc((leftSplitSize + 1) * sizeof(float));
	std::copy(toSort + leftIndex, toSort + centerIndex + 1, leftSplit);
	float* rightSplit = (float*)malloc((rightSplitSize + 1) * sizeof(float));
	std::copy((toSort + centerIndex + 1), toSort + rightIndex + 1, rightSplit);

	leftSplit[leftSplitSize] = INFINITY;
	rightSplit[rightSplitSize] = INFINITY;

	unsigned int i = 0, j = 0;
	for (unsigned int k = leftIndex; k < rightIndex+1; k++)
	{
		if (leftSplit[i] <= rightSplit[j])
		{
			toSort[k] = leftSplit[i];
			i++;
		}
		else
		{
			toSort[k] = rightSplit[j];
			j++;
		}
	}

	free(leftSplit);
	free(rightSplit);
}

void mergeSort(float toSort[], unsigned int leftIndex, unsigned int rightIndex)
{
	if (leftIndex < rightIndex)
	{
		unsigned int centerIndex = (leftIndex + rightIndex) / 2; //The indicies should never be negative, so we should always get a rounding towards zero due to truncation
		mergeSort(toSort, leftIndex, centerIndex);
		mergeSort(toSort, centerIndex + 1, rightIndex);
		merge(toSort, leftIndex, centerIndex, rightIndex);
	}
}

void mergeSort(float* toSort, unsigned int size) //For convenience
{
	mergeSort(toSort, 0, size - 1);
}