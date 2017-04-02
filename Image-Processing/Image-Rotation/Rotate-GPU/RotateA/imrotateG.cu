#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

cudaError_t launch_helper(Mat image, int *CPU_OutputArray, float* Runtimes);

int N;
int R;  //  rows 
int C;  // columns 
double ScaleFactor;
double CRA,SRA;
char *filename;

__global__ void lab5_kernel(uchar *GPU_i, uchar *GPU_o, int R, int C, double SRA, double CRA, double ScaleFactor)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int j = blockIdx.y * blockDim.y + threadIdx.y;  // column of image
 
  double o_r, o_c; 
	int r,c;
  double nr,nc;
  nr = (double)R/2 - (double)i;
  nc = (double) j - (double)C/2;
	
	o_c = CRA*nc - SRA*nr ; // col
	o_r = SRA*nc + CRA*nr; //row
	o_c = o_c * ScaleFactor; 
	o_r = o_r * ScaleFactor;
 
  r = R/2 - (int)o_r ;
  c = (int)o_c + C/2;
   
	int odx = r*C*3 +   3*c;
	int idx = i*C*3 +   3*j; 
	
	  GPU_o[odx] = GPU_i[idx];
    GPU_o[odx+1] = GPU_i[idx+1];  // no change, REPLACE THIS
    GPU_o[odx+2] = GPU_i[idx+2]; 
    		
}


int main(int argc, char *argv[])
{
	float GPURuntimes[4];		
	cudaError_t cudaStatus;
  int i ;

	if( argc != 4) {
	  printf("Usage: %s <input image> <output image> <numberOfimages>\n", argv[0]);
		exit(EXIT_FAILURE);
	}
 
	N = atoi(argv[3]);
 
 if(N< 1 || N > 30){
	  printf("Value of N %d should be less than 30 \n", argv[3]);
		exit(EXIT_FAILURE);
	}
 
 for (i = 1 ; i < N ; i ++ )
  
  {
 	
	int *CPU_OutputArray;
	double RotAngle;
  int RotDegrees;
  double Diagonal, H, V;
  
	Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
 
	if(!image.data) {
		fprintf(stderr, "Could not open or find the image.\n");
		exit(EXIT_FAILURE);
	}
	
	printf("Loaded image '%s', size = %dx%d (dims = %d).\n", argv[1], image.rows, image.cols, image.dims);
 
 	R = image.rows;
  C = image.cols;
  

      	CPU_OutputArray = (int*)malloc(3*C*R*sizeof(int));
      	if (CPU_OutputArray == NULL) {
      		fprintf(stderr, "OOPS. Can't create CPU_OutputArray using malloc() ...\n");
      		exit(EXIT_FAILURE);
      	}
      	
      	// Calculation for rotating an image
          int k; 
          
          for(k=0;k< 3*R*C; k++){
          CPU_OutputArray[k] = 0;
          }
        
        	RotDegrees = i *(360/N);
          printf("%d\n",	RotDegrees );
        	H=(double)C;
        	V=(double)R;
        	Diagonal=sqrt(H*H+V*V);
        	ScaleFactor =(C>R) ? V/Diagonal : H/Diagonal;
        	RotAngle=2*3.141592/360.000*(double)RotDegrees;
        	CRA=cos(RotAngle);
        	SRA=sin(RotAngle);
         
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
      
      
      	cudaStatus = cudaDeviceReset();
      	if (cudaStatus != cudaSuccess) {
      		fprintf(stderr, "cudaDeviceReset failed!\n");
      		free(CPU_OutputArray);
      		exit(EXIT_FAILURE);
      	}
      
      	Mat result = Mat(R, C, CV_8UC3, CPU_OutputArray);
        
        char * output_filename = argv[2];
        char fn[100]; 
       
        sprintf(fn,"dogR%03d.bmp",i);
        
      	if (!imwrite(fn, result)) {
      		fprintf(stderr, "couldn't write output to disk!\n");
      		free(CPU_OutputArray);
      		exit(EXIT_FAILURE);
      	}
       
      	printf("Saved image '%s', size = %dx%d (dims = %d).\n",
      	       fn, result.rows, result.cols, result.dims);
      
      	free(CPU_OutputArray);
 
 }
	exit(EXIT_SUCCESS);
}

cudaError_t launch_helper(Mat image, int *CPU_OutputArray, float* Runtimes)
{
	cudaEvent_t time1, time2, time3, time4; 
	int TotalGPUSize;
	uchar *GPU_idata;
	uchar *GPU_odata;

	dim3 threadsPerBlock;
	dim3 numBlocks;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);  
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}

	cudaEventCreate(&time1);
	cudaEventCreate(&time2);
	cudaEventCreate(&time3);
	cudaEventCreate(&time4);

	cudaEventRecord(time1, 0);

    TotalGPUSize = 3*R*C*sizeof(uchar);
 
	cudaStatus = cudaMalloc((void**)&GPU_idata, TotalGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
 
	cudaStatus = cudaMalloc((void**)&GPU_odata, TotalGPUSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(GPU_idata, image.data, TotalGPUSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
 
   cudaStatus = cudaMemcpy(GPU_odata,CPU_OutputArray,3*R*C, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!\n");
    goto Error;
  }


	cudaEventRecord(time2, 0);

	threadsPerBlock = dim3(8, 8);
	numBlocks = dim3(R/ threadsPerBlock.y, C/ threadsPerBlock.x);
	lab5_kernel<<<numBlocks, threadsPerBlock>>>(GPU_idata, GPU_odata, R, C, SRA, CRA, ScaleFactor);


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "error code %d (%s) launching kernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching addKernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaEventRecord(time3, 0);

	cudaStatus = cudaMemcpy(CPU_OutputArray, GPU_odata, TotalGPUSize, cudaMemcpyDeviceToHost);
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
