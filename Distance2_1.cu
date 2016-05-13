
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <cuda.h>

extern "C"

__global__ void dist2_1(float *in1, float *in2, float *out, int rows, int columns){

	int column = threadIdx.x + blockIdx.x*blockDim.x;

	if (column < columns)
	{
                 out[column] = sqrt((in1[column] - in2[0])*(in1[column] - in2[0]) +
				(in1[column + columns] - in2[1])*(in1[column + columns] - in2[1]) +
				(in1[column + 2 * columns] - in2[2])*(in1[column + 2 * columns] - in2[2]));
	}
}

