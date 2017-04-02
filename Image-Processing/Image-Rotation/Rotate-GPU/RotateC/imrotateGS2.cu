
#include <stdio.h>
#include <unistd.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

cudaError_t launch_helper(Mat image, uchar *CPU_OutputArray, float* Runtimes, int sm_d_x, int sm_d_y);

int BOX_SIZE;
int N;
int R;  //  rows 
int C;  // columns 
double RotAngle;
int RotDegrees;
double Diagonal, H, V, ScaleFactor;
double CRA,SRA;
char *filename;

#define shared_mem_limit 40000 

          
bool show_images;	

__global__ void lab6_kernel_shared(uchar *GPU_i, uchar *GPU_o, int R, int C, double SRA, double CRA, double ScaleFactor)
{
  extern __shared__ uchar shared_GPU_i[];
   
  int sm_box_height = R / gridDim.y;  
  int sm_box_width  = C*3 / gridDim.x;  
  
  int px_per_th_y = sm_box_height/blockDim.y;
  int px_per_th_x = sm_box_width/blockDim.x/3; 
  int top_row  = blockIdx.y * sm_box_height;
  int left_col = blockIdx.x * sm_box_width ;
  
  int i, j, local_r, local_c, global_offset, local_offset;
  for (i=0; i < px_per_th_y; i++) {
    for (j=0; j < px_per_th_x; j++) {
      local_r = threadIdx.y*px_per_th_y + i;
      local_c = threadIdx.x*px_per_th_x + j;
      global_offset = (top_row + local_r)*C*3+ (left_col + local_c*3);
      local_offset  = (local_r)*sm_box_width + local_c*3;	  
      shared_GPU_i[local_offset]   = GPU_i[global_offset];
      shared_GPU_i[local_offset+1] = GPU_i[global_offset+1];
      shared_GPU_i[local_offset+2] = GPU_i[global_offset+2];
    }
  }
  __syncthreads();
  
  
  for (i=0; i < px_per_th_y; i++) {
    for (j=0; j < px_per_th_x; j++) {
	
      local_r = threadIdx.y*px_per_th_y + i;
      local_c = threadIdx.x*px_per_th_x + j;
      
	    int pixCol, pixRow;
      
      pixCol = (local_c*3  + left_col)/3 ;
      pixRow = local_r  + top_row;
      
   	  
      double o_r, o_c; 
      int r,c;
      double nr,nc;
      nr = (double)R/2 - (double)pixRow;
      nc = (double)(pixCol) - (double)C/2;
	  
      o_c = CRA*nc - SRA*nr ; 
      o_r = SRA*nc + CRA*nr; 
      o_c = o_c * ScaleFactor; 
      o_r = o_r * ScaleFactor;
      
      r = (int)R/2 - (int)o_r ;
      c = (int)o_c + (int)C/2;
      
      int odx =  r*C *3 + c *3;
      int idx =  local_r*sm_box_width + (local_c*3) ;
      
      GPU_o[odx] = shared_GPU_i[idx]; 
      GPU_o[odx+1] = shared_GPU_i[idx+1];
      GPU_o[odx+2] = shared_GPU_i[idx+2];
        
    }
  }
}

 
int main(int argc, char *argv[])
{
  float GPURuntimes[4];    // run times of the GPU code
  cudaError_t cudaStatus;
  uchar *CPU_OutputArray;  // where the GPU should copy the output back to
    int shared_mem_side_x = 0;
   int shared_mem_side_y = 0;
   int i;
   int config;

  if ( argc != 5 ) {
    printf("Usage: %s <input image> <output image><NumberOfimageRotation><Config type>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  
  N = atoi(argv[3]);
  
  if(N< 1 || N > 30){
	  printf("Value of N %d should be less than 30 \n", argv[3]);
		exit(EXIT_FAILURE);
	}
	
  config = atoi(argv[4]);
  
  if(config < 1 || config > 4){
	  printf("Value of configuration type %d should be between 1 to 4 \n", argv[4]);
		exit(EXIT_FAILURE);
	}
	
  switch(config){
  case 1: BOX_SIZE = 16;shared_mem_side_x =192 ; shared_mem_side_y =160 ; break;
  case 2: BOX_SIZE = 16;shared_mem_side_x =96 ; shared_mem_side_y =96 ; break;
  case 3: BOX_SIZE = 32;shared_mem_side_x =192 ;shared_mem_side_y =160 ; break;
  case 4: BOX_SIZE = 32;shared_mem_side_x =96 ;shared_mem_side_y =96 ; break;
  default: 
    printf("Usage: %s <input image> <output image><NumberOfimageRotation><Config type>\n", argv[0]);
    printf("where 'show images' is 0 or 1, and block size 16 means 16x16 threads per block\n");
    exit(EXIT_FAILURE);
  
  }
  
  
  Mat image;
  image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!image.data) {
    fprintf(stderr, "Could not open or find the image.\n");
    exit(EXIT_FAILURE);
  }
  printf("Loaded image '%s', size = %dx%d (dims = %d).\n", argv[1], image.rows, image.cols, image.dims);
  R = image.rows;
  C = image.cols;
   
  CPU_OutputArray = (uchar*)malloc(3*R*C*sizeof(uchar));
          if (CPU_OutputArray == NULL) {
            fprintf(stderr, "OOPS. Can't create CPU_OutputArray using malloc() ...\n");
            exit(EXIT_FAILURE);
          }


  for( i = 1; i < N ; i++)
  
  {       
          RotDegrees = i*(360/N);
          H =(double)C;
          V =(double)R;
          Diagonal=sqrt(H*H+V*V);
          ScaleFactor =(C>R) ? V/Diagonal : H/Diagonal;
          RotAngle=2*3.141592/360.000*(double)RotDegrees;
          CRA=cos(RotAngle);
          SRA=sin(RotAngle);
          
          //shared_mem_side_x = 96; 
          //shared_mem_side_y = 96; 

	 int k; 

	 for(k=0;k< 3*R*C; k++){
	 CPU_OutputArray[k] = 0;
	 }
          
          /*shared_mem_side_x = (int)(sqrt(shared_mem_limit) - fmod(sqrt(shared_mem_limit), BOX_SIZE));
          while ((C*3 % shared_mem_side_x != 0) && (shared_mem_side_x % 3 != 0) && (R %  shared_mem_side_x != 0)) {
            shared_mem_side_x -= BOX_SIZE;
            if (shared_mem_side_x % BOX_SIZE != 0) {
              shared_mem_side_x -= BOX_SIZE;
            }
            if (shared_mem_side_x <= 0) { break; }
          }
          if (shared_mem_side_x <= 0) {
            fprintf(stderr, "Unable to find good way to break up the image; exiting.\n");
            exit(EXIT_FAILURE);
          }
          else {
            printf("Each block will process %i col RGB pixels (%f%% of shared memory target).\n",
        	  shared_mem_side_x,
        	   100*(float)(shared_mem_side_x*shared_mem_side_x)/shared_mem_limit);
          }
          
            shared_mem_side_y = (int)(sqrt(shared_mem_limit) - fmod(sqrt(shared_mem_limit), BOX_SIZE));
          while ((R %  shared_mem_side_y != 0)) {
             shared_mem_side_y -= BOX_SIZE;
            if ( shared_mem_side_y % BOX_SIZE != 0) {
               shared_mem_side_y -= BOX_SIZE;
            }
            if ( shared_mem_side_y <= 0) { break; }
          }
          if ( shared_mem_side_y <= 0) {
            fprintf(stderr, "Unable to find good way to break up the image; exiting.\n");
            exit(EXIT_FAILURE);
          }
          else {
            printf("Each block will process %i row pixels (%f%% of shared memory target).\n",
        	    shared_mem_side_y,
        	   100*(float)( shared_mem_side_y* shared_mem_side_y)/shared_mem_limit);
          }*/
          

         
         
          cudaStatus = launch_helper(image, CPU_OutputArray, GPURuntimes,shared_mem_side_x,shared_mem_side_y);
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
        
  }
        free(CPU_OutputArray);
 
          exit(EXIT_SUCCESS);
}

cudaError_t launch_helper(Mat image, uchar *CPU_OutputArray, float* Runtimes, int sm_d_x,int sm_d_y)
{
  cudaEvent_t time1, time2, time3, time4;
  int TotalGPUSize;  
  uchar *GPU_idata;
  uchar *GPU_odata;

  dim3 threadsPerBlock;
  dim3 numBlocks;
  int shared_mem_size;
  
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

  TotalGPUSize = 3 * R * C * sizeof(uchar);
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

  threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);

  numBlocks = dim3( C*3/sm_d_x, R/sm_d_y );

  shared_mem_size = sm_d_x*sm_d_y*sizeof(uchar);
  lab6_kernel_shared <<<numBlocks, threadsPerBlock, shared_mem_size>>> (GPU_idata, GPU_odata, R, C, SRA, CRA, ScaleFactor);

  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess)
    {
      fprintf(stderr, "error code %d (%s) launching kernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
      goto Error;
    }

  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching addKernel!\n",
	    cudaStatus, cudaGetErrorString(cudaStatus));
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
