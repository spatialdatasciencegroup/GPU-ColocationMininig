#include <iostream>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "colocationFinder.h"
using namespace std;


int main(int argc, char**argv) {
	int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
		printf("cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
		printf("\nFAILED\n");
		return -1;
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("There is no device supporting CUDA\n");
		return -1;
	}

	cudaDeviceProp deviceProp;
	for (Integer i = 0; i < deviceCount; ++i)
	{
		Integer dev = i;
		cudaGetDeviceProperties(&deviceProp, dev);
	}
	if (deviceCount == 1)
	{
		// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
		if (deviceProp.major == 9999 && deviceProp.minor == 9999)
		{
			printf("There is no device supporting CUDA.\n");
			return -1;
		}
	}
	cudaSetDevice(0);
	//printf("%d Devices Found\n", deviceCount);
	colocationFinder* oColocationFinder = NULL;
	oColocationFinder = new colocationFinder();
    

	oColocationFinder->Begin(argc,argv);


	return 0;
}
