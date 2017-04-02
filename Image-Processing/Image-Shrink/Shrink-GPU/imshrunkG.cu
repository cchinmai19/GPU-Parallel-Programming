// ECE 406 Lab 5, Fall 2015

#include <stdio.h>

// CUDA stuff:
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// OpenCV stuff (note: C++ not C):
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

cudaError_t launch_helper(Mat image, int *CPU_OutputArray, float* Runtimes);

int M;  // number of rows in image
int N;  // number of columns in image

unsigned int shrinkRatiox,shrinkRatioy;
unsigned int x,y;

// These come from CLI arguments:

__global__ void lab5_kernel(uchar *GPU_i, uchar *GPU_o, int M, int N, int x, int y, int shrinkRatiox, int shrinkRatioy)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int j = blockIdx.y * blockDim.y + threadIdx.y;  // column of image
	int odx = i*x*3 +   3*j;  // which pixel in full 1D array
	int idx=i*shrinkRatioy*N*3+3*j*shrinkRatiox;
  if(odx<x*y*3 && idx<M*N*3) {
  
	GPU_o[odx] = GPU_i[idx];
    GPU_o[odx+1] = GPU_i[idx+1];  // no change, REPLACE THIS
    GPU_o[odx+2] = GPU_i[idx+2];  // no change, REPLACE THIS
  }
}

int main(int argc, char *argv[])
{
	float GPURuntimes[4];		// run times of the GPU code
	cudaError_t cudaStatus;
	int *CPU_OutputArray;		// where the GPU should copy the output back to

	if( argc != 5) {
	  printf("Usage: %s <input image> <output image> <shrinkRatiox> <shrinkRatioy>\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	shrinkRatiox=atoi(argv[3]);
	shrinkRatioy=atoi(argv[4]);

	Mat image;
  image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
 
	// we could load it as CV_LOAD_IMAGE_COLOR, but we don't want to worry about that extra dimension
	if(! image.data ) {
		fprintf(stderr, "Could not open or find the image.\n");
		exit(EXIT_FAILURE);
	}
	printf("Loaded image '%s', size = %dx%d (dims = %d).\n", argv[1], image.rows, image.cols, image.dims);

	// Set up global variables based on image size:
	M = image.rows;
	N = image.cols;
	
	x=(N/shrinkRatiox);
	y=(M/shrinkRatioy);
  
  printf("\n%d %d %d %d\n", x,y,M,N);
  

	CPU_OutputArray = (int*)malloc(3*x*y*sizeof(int));
  if (CPU_OutputArray == NULL) {
		fprintf(stderr, "OOPS. Can't create CPU_OutputArray using malloc() ...\n");
		exit(EXIT_FAILURE);
	}

	// Run it:
	cudaStatus = launch_helper(image, CPU_OutputArray, GPURuntimes);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "launch_helper failed!\n");
		free(CPU_OutputArray);
		exit(EXIT_FAILURE);
	}

	printf("-----------------------------------------------------------------\n");
	printf("Tfr CPU->GPU = %5.2f ms ... \nExecution = %5.2f ms ... \nTfr GPU->CPU = %5.2f ms   \n Total=%5.2f ms\n",
			GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
	printf("-----------------------------------------------------------------\n");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		free(CPU_OutputArray);
		exit(EXIT_FAILURE);
	}

	// Display the output image:
	//Mat result = Mat(y, x, CV_8UC1, CPU_OutputArray);
	Mat result = Mat(y, x, CV_8UC3, CPU_OutputArray);
  // and save it to disk:
	string output_filename = argv[2];
	if (!imwrite(output_filename, result)) {
		fprintf(stderr, "couldn't write output to disk!\n");
		free(CPU_OutputArray);
		exit(EXIT_FAILURE);
	}
	printf("Saved image '%s', size = %dx%d (dims = %d).\n",
	       output_filename.c_str(), result.rows, result.cols, result.dims);

	free(CPU_OutputArray);
	exit(EXIT_SUCCESS);
}

// Helper function for launching a CUDA kernel (including memcpy, timing, etc.):
cudaError_t launch_helper(Mat image, int *CPU_OutputArray, float* Runtimes)
{
	cudaEvent_t time1, time2, time3, time4;
	int TotalOutputGPUSize;  // total size of 1 image (i.e. input or output) in bytes
	int TotalGPUSize;
	uchar *GPU_idata;
	uchar *GPU_odata;
	// Note that we could store GPU_i and GPU_o as 2D arrays instead of 1D...
	// it would make indexing simpler, but could complicate memcpy.

	dim3 threadsPerBlock;
	dim3 numBlocks;

	// Choose which GPU to run on; change this on a multi-GPU system.
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);  // use the first GPU (not necessarily the fastest)
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	cudaEventCreate(&time1);
	cudaEventCreate(&time2);
	cudaEventCreate(&time3);
	cudaEventCreate(&time4);

	cudaEventRecord(time1, 0);

  
	TotalGPUSize = 3*M * N * sizeof(uchar);
   TotalOutputGPUSize = 3*x * y * sizeof(uchar);
 
	cudaStatus = cudaMalloc((void**)&GPU_idata, TotalGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
 
	cudaStatus = cudaMalloc((void**)&GPU_odata, TotalOutputGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPU_idata, image.data, TotalGPUSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaEventRecord(time2, 0);

	// Launch a kernel on the GPU with one thread for each pixel.
	threadsPerBlock = dim3(8, 8);
	numBlocks = dim3(y / threadsPerBlock.y, x / threadsPerBlock.x);
	lab5_kernel<<<numBlocks, threadsPerBlock>>>(GPU_idata, GPU_odata, M, N, x, y, shrinkRatiox, shrinkRatioy);

	// Check for errors immediately after kernel launch.
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "error code %d (%s) launching kernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching addKernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaEventRecord(time3, 0);

	// Copy output (results) from GPU buffer to host (CPU) memory.
	cudaStatus = cudaMemcpy(CPU_OutputArray, GPU_odata, TotalOutputGPUSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaEventRecord(time4, 0);
	cudaEventSynchronize(time1);
	cudaEventSynchronize(time2);
	cudaEventSynchronize(time3);
	cudaEventSynchronize(time4);

	float totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime;

	cudaEventElapsedTime(&totalTime, time1, time4);
	cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
	cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
	cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);

	Runtimes[0] = totalTime;
	Runtimes[1] = tfrCPUtoGPU;
	Runtimes[2] = kernelExecutionTime;
	Runtimes[3] = tfrGPUtoCPU;

	Error:
	cudaFree(GPU_odata);
	cudaFree(GPU_idata);
	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);

	return cudaStatus;
}
