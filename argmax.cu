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
extern "C" __global__ void argmax(const double *A, int *B, int rows, int cols) {
	extern __shared__ int s[];
	int *maxindexes = s;
	double *maxvalues = (double*)&maxindexes[blockDim.x*blockDim.y];

	unsigned int tidx = threadIdx.x;
	unsigned int tidy = threadIdx.y;

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	maxindexes[tidx + tidy*blockDim.x] = (x < rows && y < cols) ? tidx : 0;
	maxvalues[tidx + tidy*blockDim.x] = (x < rows && y < cols) ? A[x + rows*y] : 0;

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tidx < s  && x + s < rows && y < cols)
		{
			if (maxvalues[tidx + tidy*blockDim.x +s] > maxvalues[tidx + tidy*blockDim.x]) {
				maxvalues[tidx + tidy*blockDim.x] = maxvalues[tidx + tidy*blockDim.x + s];
				maxindexes[tidx + tidy*blockDim.x] = maxindexes[tidx + tidy*blockDim.x + s];
			}
		}

		__syncthreads();
	}

	if (tidx == 0) {
		B[tidy] = maxindexes[tidy*blockDim.x];
	}
}
