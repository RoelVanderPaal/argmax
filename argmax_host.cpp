#include <stdio.h>
#include <iostream>
#include <cuda.h>

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction argmax;

int *h_in_indexes;
double *h_in_values;
int *h_out_indexes;
double *h_out_values;
CUdeviceptr d_in_indexes;
CUdeviceptr d_in_values;
CUdeviceptr d_out_indexes;
CUdeviceptr d_out_values;


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

int main() {
	checkCudaErrors(cuInit(0));
	checkCudaErrors(cuDeviceGet(&cuDevice, 0));
	checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));
	checkCudaErrors(cuModuleLoad(&cuModule, "data/argmax64.ptx"));
	checkCudaErrors(cuModuleGetFunction(&argmax, cuModule, "argmax"));

	int rows = 16;
	int cols = 6;
	int N = rows*cols;
	int rowsPerBlock = 256;
	int colsPerBlock = 1;


	int blockRows = (rows + rowsPerBlock - 1) / rowsPerBlock;
	int blockCols = (cols + colsPerBlock - 1) / colsPerBlock;

	h_in_values = (double *)malloc(N*sizeof(double));
	h_in_indexes = (int *)malloc(N*sizeof(int));
	srand(125);
	for (int col = 0; col < cols; col++) {
		for (int row = 0; row < rows; row++) {
			h_in_values[row + col*rows] = (rand()) - 1000;
			h_in_indexes[row + col*rows] = row;
		}
	}
	h_out_indexes = (int *)malloc(blockRows*cols*sizeof(int));
	h_out_values = (double *)malloc(blockRows*cols*sizeof(double));

	checkCudaErrors(cuMemAlloc(&d_in_values, N*sizeof(double)));
	checkCudaErrors(cuMemcpyHtoD(d_in_values, h_in_values, N*sizeof(double)));
	checkCudaErrors(cuMemAlloc(&d_in_indexes, N*sizeof(int)));
	checkCudaErrors(cuMemcpyHtoD(d_in_indexes, h_in_indexes, N*sizeof(int)));
	checkCudaErrors(cuMemAlloc(&d_out_indexes, blockRows*cols*sizeof(int)));
	checkCudaErrors(cuMemAlloc(&d_out_values, blockRows*cols*sizeof(double)));

	void *args[] = { &d_in_indexes, &d_in_values, &d_out_indexes, &d_out_values, &rows, &cols };

	std::cout << blockRows << " " << blockCols << "\n\n";

	checkCudaErrors(cuLaunchKernel(argmax, blockRows, blockCols, 1,
		rowsPerBlock, colsPerBlock, 1,
		rowsPerBlock*colsPerBlock*(sizeof(int) + sizeof(double)),
		NULL, args, NULL));

	checkCudaErrors(cuMemcpyDtoH(h_out_indexes, d_out_indexes, blockRows*cols*sizeof(int)));
	checkCudaErrors(cuMemcpyDtoH(h_out_values, d_out_values, blockRows*cols*sizeof(double)));
	for (int col = 0; col < cols; col++) {
		for (int row = 0; row < rows; row++) {
			std::cout << h_in_values[row + rows*col] << "\n";
		}
		for (int blockRow = 0; blockRow < blockRows; blockRow++){
			std::cout << h_out_indexes[col*blockRows + blockRow] << " " << h_out_values[col*blockRows + blockRow] << "\n";
		}
		std::cout << "\n";
	}

	checkCudaErrors(cuMemFree(d_in_indexes));
	checkCudaErrors(cuMemFree(d_in_values));
	checkCudaErrors(cuMemFree(d_out_indexes));
	checkCudaErrors(cuMemFree(d_out_values));
	free(h_in_indexes);
	free(h_in_values);
	free(h_out_indexes);

	free(h_out_values);
	checkCudaErrors(cuModuleUnload(cuModule));
	checkCudaErrors(cuCtxDestroy(cuContext));
	printf("gedaan\n");
}