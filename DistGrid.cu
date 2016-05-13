
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <stdio.h>
extern "C"

__global__ void distGrid(float *in1, float *in2, float *out, int columns1, int columns2  )
{

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < columns1)
	{
		for (int i = 0; i < columns2; i++)


		out[idx*i] = sqrt((in1[idx] - in2[i])*(in1[idx] - in2[i]) +
		(in1[idx + columns1] - in2[i + columns2])*(in1[idx + columns1] - in2[i + columns2]) +
		(in1[idx + 2 * columns1] - in2[i + columns2 * 2])*(in1[idx + 2 * columns1] - in2[i + columns2*2]));
	}
}
