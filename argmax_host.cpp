#include <stdio.h>
#include <iostream>
#include <cuda.h>

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction argmax;

double *h_A;
int *h_B;
CUdeviceptr d_A;
CUdeviceptr d_B;


// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
	if (CUDA_SUCCESS != err)
	{
		fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
			err, err, file, line);
		exit(EXIT_FAILURE);
	}
}

void RandomInit(double *data, int n)
{
	srand(123);
	for (int i = 0; i < n; ++i)
	{
		data[i] = rand() / 1.2;
	}
}



int main() {
	checkCudaErrors(cuInit(0));
	checkCudaErrors(cuDeviceGet(&cuDevice, 0));
	checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));
	checkCudaErrors(cuModuleLoad(&cuModule, "data/argmax64.ptx"));
	checkCudaErrors(cuModuleGetFunction(&argmax, cuModule, "argmax"));

	int rows = 4;
	int cols = 2;
	int N = rows*cols;
	h_A = (double *)malloc(N*sizeof(double));
	RandomInit(h_A, N);
	h_B = (int *)malloc(cols*sizeof(int));

	checkCudaErrors(cuMemAlloc(&d_A, N*sizeof(double)));
	checkCudaErrors(cuMemAlloc(&d_B, cols*sizeof(int)));
	checkCudaErrors(cuMemcpyHtoD(d_A, h_A, N*sizeof(double)));


	int threadsPerBlock = 16;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	void *args[] = { &d_A, &d_B, &rows, &cols};

	checkCudaErrors(cuLaunchKernel(argmax, (rows + threadsPerBlock - 1) / threadsPerBlock, (cols + threadsPerBlock - 1) / threadsPerBlock, 1,
		threadsPerBlock, threadsPerBlock, 1,
		threadsPerBlock*threadsPerBlock*(sizeof(int) + sizeof(double)),
		NULL, args, NULL));

	checkCudaErrors(cuMemcpyDtoH(h_B, d_B, cols*sizeof(int)));
	for (int col = 0; col < cols; col++) {
		for (int row = 0; row < rows; row++) {
			std::cout << h_A[row+rows*col] << "\n";
		}
		std::cout << h_B[col] << "\n";
	}

	checkCudaErrors(cuMemFree(d_A));
	checkCudaErrors(cuMemFree(d_B));
	free(h_A);
	free(h_B);
	checkCudaErrors(cuModuleUnload(cuModule));
	checkCudaErrors(cuCtxDestroy(cuContext));
	printf("gedaan\n");
}