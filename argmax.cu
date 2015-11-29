/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */

// Device code
extern "C" __global__ void argmax(const int *in_indexes, const double *in_values, int *out_indexes, double *out_values, int rows, int cols) {
	extern __shared__ int s[];
	int *maxindexes = s;
	double *maxvalues = (double*)&maxindexes[blockDim.x*blockDim.y];

	unsigned int tidx = threadIdx.x;
	unsigned int tidy = threadIdx.y;
	unsigned int i = tidx + tidy*blockDim.x;

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	maxindexes[i] = (x < rows && y < cols) ? in_indexes[x + rows*y] : 0;
	maxvalues[i] = (x < rows && y < cols) ? in_values[x + rows*y] : 0;

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tidx < s  && x + s < rows && y < cols)
		{
			if (maxvalues[i + s] > maxvalues[i]) {
				maxvalues[i] = maxvalues[i + s];
				maxindexes[i] = maxindexes[i + s];
			}
		}

		__syncthreads();
	}

	if (tidx == 0 && y < cols) {
		out_indexes[gridDim.x*(tidy+ (blockIdx.y*blockDim.y)) + blockIdx.x] = maxindexes[blockDim.x * tidy];
		out_values[gridDim.x*(tidy + (blockIdx.y*blockDim.y)) + blockIdx.x] = maxvalues[blockDim.x * tidy];
	}
}
