//ECE 406 final project
// Sarang Lele, Karthik Dinesh, Chinmai
//Cuda support programs

#include <stdio.h>
#include <assert.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    // We should be free()ing CPU+GPU memory here, but we're relying on the OS
    // to do it for us.
    cudaDeviceReset();
    assert(result == cudaSuccess);
  }
  return result;
}

void WriteNumbers(char* filename, float *features, int row, int col, int numbins)
{
	int i,j;
	//unsigned long int numbers[8192];
	FILE* f = fopen(filename, "w");
	if(f == NULL)
	{
		printf("\n\n%s NOT FOUND\n\n",filename);
		exit(1);
	}
	
	for (i=0;i<row*col;i++)
	{
		for(j=0;j<numbins;j++) {
      			if(numbins==4) fprintf(f, "%f ", features[i*numbins+j]);
		  	else fprintf(f, "%e ", features[i*numbins+j]);
		}
   	fprintf(f,"\n");
	}
}
