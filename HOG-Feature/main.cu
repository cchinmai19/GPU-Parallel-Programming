// ECE 406 Final Project
// Sarang Lele, Karthik Dinesh, Chinmai

#include <stdio.h>
#include <string.h>

// CUDA stuff:
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// OpenCV stuff (note: C++ not C):
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//other support
#include "HogSupport.h"
#define BOX_SIZE 8
cudaError_t launch_helper(float* Runtimes);

struct HogProp hp;
struct DisplayProp dp;
uchar * CPU_InputArray, * CPU_OutputArray;
float *CPU_CellArray,*CPU_FeatureArray, *CPU_Hist;

cudaStream_t stream[2];

using namespace cv;

__global__ void Cal_kernel(uchar *GPU_i, int *Orientation,float *Gradient, uchar *DisplayOrientation, HogProp hp)
{
 	int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image
   
  float ang,displayang;
	float gx,gy;
  int idx = i*hp.ImgCol + j;
  int idx_prev= (i-1)*hp.ImgCol + j;
  int idx_next= (i+1)*hp.ImgCol + j;
 
  if(i>0 && i < hp.ImgRow-1 && j >0 && j < hp.ImgCol-1){
   	gx=(float)(GPU_i[idx-1]-GPU_i[idx+1]);
    gy=(float)(GPU_i[idx_prev]-GPU_i[idx_next]);
    
     Gradient[idx]=sqrtf(gx*gx+gy*gy);
     ang= atan2f(gy,gx);
     
     if(ang<0) {
       displayang=8*(ang+PI);
     }
     else displayang=8*ang;
     
     if(displayang<PI | displayang>7*PI)          DisplayOrientation[idx]=0;
     else if(displayang>=PI & displayang<3*PI)    DisplayOrientation[idx]=1;
     else if(displayang>=3*PI & displayang<5*PI)  DisplayOrientation[idx]=2;
     else                                         DisplayOrientation[idx]=3;
          
     if (ang<0){
       if(hp.Orientation==0) { ang = ang+ PI; }
       else { ang = 2*PI+ang; }
     }
     
     if(hp.Orientation==0) ang=(hp.NumBins)*ang/PI;
     else ang=(hp.NumBins)*ang/(2*PI);
     
     Orientation[idx]=(int)ang;
     //GPU_o[idx] = (uchar) (DisplayOrientation[idx]);
     
  }

}

__global__ void Cell_kernel(float *histogram, int *Orientation,float *Gradient, HogProp hp)
{
 	int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image
   
  int idx = i*hp.ImgCol*hp.CellSize + j*hp.CellSize;
  int idcell = i*hp.CellCol*hp.NumBins + j*hp.NumBins;
  int current_i,m,n;
  //int idx_next= (i+1)*hp.ImgCol + j;
  
  if(i<hp.CellRow & j<hp.CellCol) {
    for (m=0;m<hp.CellSize;m++) {
      current_i=idx+m*hp.ImgCol;
      for (n=0;n<hp.CellSize;n++) {
        histogram[idcell+Orientation[current_i+n]]+=Gradient[current_i+n];
      }
    }
  }
}

__global__ void Block_kernel(float *FinalFeatures, float *histogram, HogProp hp)
{
 	int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image
  
  int step=hp.BlockSize-hp.BlockOverlap;
  int idblock = i*hp.BlockCol*hp.FeatureSize + j*hp.FeatureSize;
  int idcell = i*hp.CellCol*step*hp.NumBins + j*step*hp.NumBins;
  int current_i,current_j,m,n;
  float average=0.000000001;
  int horz=hp.BlockSize*hp.NumBins;
  //int idx_next= (i+1)*hp.ImgCol + j;
  
  if(i<hp.BlockRow & j<hp.BlockCol) {
    for (m=0;m<hp.BlockSize;m++) {
      current_i=idcell+m*hp.CellCol*hp.NumBins;
      for (n=0;n<horz;n++) {
        average=average+histogram[current_i+n];
      }
    }
  }
  
  if(i<hp.BlockRow & j<hp.BlockCol) {
    for (m=0;m<hp.BlockSize;m++) {
      current_i=idcell+m*hp.CellCol*hp.NumBins;
      current_j=idblock+m*hp.CellCol;
      for (n=0;n<horz;n++) {
        FinalFeatures[current_j+n]=histogram[current_i+n]/average;
      }
    }
  }
}

__global__ void Display_Cell_kernel(float* Displayhistogram, float *TempDisplayhistogram, uchar *DisplayOrientation,float *Gradient, DisplayProp dp)
{
 	int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image
   
  int idx = i*dp.ImgCol + j*dp.CellSize;
  int idxtemp = i*dp.CellCol*dp.NumBins*dp.CellSize + j*dp.NumBins;
  int idcell = i*dp.CellCol*dp.NumBins + j*dp.NumBins;
  int n;
  int temp_rowsize=dp.CellCol*dp.NumBins;
  //float avg;
  float max1,max2,avg;
  //int idx_next= (i+1)*hp.ImgCol + j;
  
  if(i<dp.HorzCells & j<dp.CellCol) {
    TempDisplayhistogram[idcell]=0; TempDisplayhistogram[idcell+1]=0; TempDisplayhistogram[idcell+2]=0; TempDisplayhistogram[idcell+3]=0;
    for (n=0;n<dp.CellSize;n++) {
      TempDisplayhistogram[idcell+DisplayOrientation[idx+n]]+=Gradient[idx+n];
    }
  }
  
  __syncthreads();
  
  if(i<dp.CellRow) {
    
    for(n=0;n<dp.CellSize;n++) {
      Displayhistogram[idcell]+=TempDisplayhistogram[idxtemp+n*temp_rowsize];
      Displayhistogram[idcell+1]+=TempDisplayhistogram[idxtemp+n*temp_rowsize+1];
      Displayhistogram[idcell+2]+=TempDisplayhistogram[idxtemp+n*temp_rowsize+2];
      Displayhistogram[idcell+3]+=TempDisplayhistogram[idxtemp+n*temp_rowsize+3];
    }
    
    if(Displayhistogram[idcell]>Displayhistogram[idcell+1]) {max1=Displayhistogram[idcell];}   else {max1=Displayhistogram[idcell+1];}
    if(Displayhistogram[idcell+2]>Displayhistogram[idcell+3]) {max2=Displayhistogram[idcell+2];} else {max2=Displayhistogram[idcell+3];}
    if(max2>max1) max1=max2;
    avg=max1/8;
    //avg=(Displayhistogram[idcell+3]+Displayhistogram[idcell+2]+Displayhistogram[idcell+1]+Displayhistogram[idcell])/8;
    //avg=1;
    if(Displayhistogram[idcell+3]>=0) Displayhistogram[idcell+3]=Displayhistogram[idcell+3]/avg; else Displayhistogram[idcell+3]=0;
    if(Displayhistogram[idcell+2]>=0) Displayhistogram[idcell+2]=Displayhistogram[idcell+2]/avg; else Displayhistogram[idcell+2]=0;
    if(Displayhistogram[idcell+1]>=0) Displayhistogram[idcell+1]=Displayhistogram[idcell+1]/avg; else Displayhistogram[idcell+1]=0;
    if(Displayhistogram[idcell]>=0) Displayhistogram[idcell]=Displayhistogram[idcell]/avg; else Displayhistogram[idcell]=0;
  }
}


__global__ void display_kernel(float *Displayhistogram, uchar *GPU_odata, DisplayProp dp)
{
 	int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image
  int k = threadIdx.z;
   
  int idx = i*dp.CellCol*4 + j*4+k;
  int idcell = i*dp.DisplayCellSize*dp.DisplayImgCol + j*dp.DisplayCellSize;
  int m;
  int temp=(int)Displayhistogram[idx];
  int tempid;
  
  tempid=idcell+8+8*dp.DisplayImgCol;
  for(m=1;m<temp ;m++) {
    if(k==0) {
      GPU_odata[tempid+m]=255; GPU_odata[tempid-m]=255;
    }else if(k==1) {
      GPU_odata[tempid+m-m*dp.DisplayImgCol]=255; GPU_odata[tempid-m+m*dp.DisplayImgCol]=255;
    }else if(k==2) {
      GPU_odata[tempid-m*dp.DisplayImgCol]=255; GPU_odata[tempid+m*dp.DisplayImgCol]=255;
    }else {
      GPU_odata[tempid+m+m*dp.DisplayImgCol]=255; GPU_odata[tempid+m+m*dp.DisplayImgCol]=255;
    }
  }
  if(k==0) GPU_odata[tempid]=255;
}
//###################################################################################################################################################
//------------------------------------------------------------main function--------------------------------------------------------------------------
//###################################################################################################################################################
int main(int argc, char *argv[]) {
	
	//-------------------------------------------------------------variables-------------------------------------------------------------------------
	//int i;
  float GPURuntimes[4];
	//===============================================================================================================================================
	
	//--------------------------------------------------Input Parameter error chec-------------------------------------------------------------------
	switch (argc){
		case 3 : hp.CellSize=8; 			hp.BlockSize=2; 			hp.BlockOverlap=1; 					  hp.NumBins=9; 			hp.Orientation=0; 			  break;
		case 4 : hp.CellSize=atoi(argv[3]); hp.BlockSize=2; 			hp.BlockOverlap=1; 			   		  hp.NumBins=9; 			hp.Orientation=0; 			  break;
		case 5 : hp.CellSize=atoi(argv[3]); hp.BlockSize=atoi(argv[4]); hp.BlockOverlap=ceil(hp.BlockSize/2); hp.NumBins=9; 			hp.Orientation=0; 			  break;
		case 6 : hp.CellSize=atoi(argv[3]); hp.BlockSize=atoi(argv[4]); hp.BlockOverlap=atoi(argv[5]); 		  hp.NumBins=9; 			hp.Orientation=0; 			  break;
		case 7 : hp.CellSize=atoi(argv[3]); hp.BlockSize=atoi(argv[4]); hp.BlockOverlap=atoi(argv[5]); 		  hp.NumBins=atoi(argv[6]); hp.Orientation=0; 			  break;
		case 8 : hp.CellSize=atoi(argv[3]); hp.BlockSize=atoi(argv[4]); hp.BlockOverlap=atoi(argv[5]); 		  hp.NumBins=atoi(argv[6]); hp.Orientation=atoi(argv[7]); break;
		default: printf("\n\nUsage: hogfeature <inputimage> <output image> <Cell Size> <Block Size> <Block Overlap> <Number of Bins> <Orintation>");
		printf("\n\nExample: hogfeature infilename.bmp outname.bmp 8 2 1 9 0\n");
		printf("\nNumber of input parameters must be between 2 and 7\n");
		return 0;
	}
	
	if(hp.CellSize<2 | hp.CellSize> 32) {
		printf("\n\nCellSize = %d is invalid",hp.CellSize);
		printf("\n Cell Size can be an integer between 2 and 32\n");
		exit(EXIT_FAILURE);
	}
	
	if(hp.BlockSize<0 | hp.BlockSize> 8) {
		printf("\n\nBlockSize = %d is invalid",hp.BlockSize);
		printf("\n Block Size can be an integer between 1 and 8\n");
		exit(EXIT_FAILURE);
	}

	if(hp.BlockOverlap<0 | hp.BlockOverlap> hp.BlockSize-1) {
		printf("\n\nBlockSize = %d is invalid",hp.BlockOverlap);
		printf("\n Block overlap can be an integer between 0 and %d\n",hp.BlockSize-1);
		exit(EXIT_FAILURE);
	}
	
	if(hp.NumBins<4 | hp.NumBins> 180) {
		printf("\n\nNumBins = %d is invalid",hp.NumBins);
		printf("\nNumber of bins can be an integer between 0 and 180\n");
		exit(EXIT_FAILURE);
	}
	
	if(hp.Orientation<0 | hp.Orientation>1) {
		printf("\n\nOrientation = %d is invalid",hp.Orientation);
		printf("\n Orientation can be either 0 or 1\n");
		exit(EXIT_FAILURE);
	}
 
  printf("\n\nPARAMETERS:\n\n");
  printf("CellSize=%d, BlockSize=%d, BlockOverlap=%d, NumBins=%d, Orientation=%d\n",hp.CellSize,hp.BlockSize,hp.BlockOverlap,hp.NumBins,hp.Orientation);
	//===============================================================================================================================================

	//----------------------------------------------------------------Load Image---------------------------------------------------------------------
	Mat image;	// see http://docs.opencv.org/modules/core/doc/basic_structures.html#mat
	image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE); //Load Grayscale image
	
	if(! image.data ) {
		fprintf(stderr, "Could not open or find the image.\n");
		exit(EXIT_FAILURE);
	}
	printf("Loaded image '%s', size = %dx%d (dims = %d).\n\n", argv[1], image.rows, image.cols, image.dims);
	hp.ImgRow=image.rows;
	hp.ImgCol=image.cols;
  hp.ImgSize=hp.ImgRow*hp.ImgCol;
  hp.CellRow=floor(image.rows/hp.CellSize);
  hp.CellCol=floor(image.cols/hp.CellSize);
  hp.TotalCells=hp.CellRow*hp.CellCol;
	hp.BlockRow=(hp.CellRow-hp.BlockSize+1)/(hp.BlockSize-hp.BlockOverlap);
  hp.BlockCol=(hp.CellCol-hp.BlockSize+1)/(hp.BlockSize-hp.BlockOverlap);
  hp.TotalBlocks=hp.BlockRow*hp.BlockCol;
  hp.FeatureSize=hp.NumBins*hp.BlockSize*hp.BlockSize;
  printf("----------------------------------IMAGE DIVIDED INTO CELL HISTOGRAM----------------\n");
  printf("\nCell_rows = %d, Cell_columns = %d, Total_cells = %d\n",hp.CellRow,hp.CellCol,hp.TotalCells);
	printf("\nBlock_rows = %d, Block_columns = %d, Total_blocks = %d\n",hp.BlockRow,hp.BlockCol,hp.TotalBlocks);
  printf("\nfeaturesize=%d\n",hp.FeatureSize);
  printf("-----------------------------------------------------------------------------------\n\n");
  
  dp.ImgRow=hp.ImgRow;
  dp.ImgCol=hp.ImgCol;
  dp.ImgSize=hp.ImgSize;
  dp.CellRow=32;
  dp.CellSize=dp.ImgRow/dp.CellRow;
  dp.CellCol=dp.ImgCol/dp.CellSize;
  dp.TotalCells=dp.CellRow*dp.CellCol;
  dp.NumBins=4;
  dp.HorzCellsTotal=dp.CellSize*dp.TotalCells;
  dp.HorzCells=dp.CellSize*dp.CellRow;
  
  dp.DisplayCellSize=17;
  dp.DisplayImgRow=dp.DisplayCellSize*dp.CellRow;
  dp.DisplayImgCol=dp.DisplayCellSize*dp.CellCol;
  dp.DisplayImgSize=dp.DisplayImgCol*dp.DisplayImgRow;
  printf("----------------------IMAGE DIVIDED INTO CELL HISTOGRAM FOR DISPLAYING-------------\n");
  printf("\nCell_rows = %d, Cell_columns = %d, Total_cells=%d, Cell_size=%d, Horz_cells=%d\n",dp.CellRow,dp.CellCol,dp.TotalCells,dp.CellSize,dp.HorzCells);
  printf("\nDisplay_rows = %d, Display_columns = %d, Total_pixels=%d, Cell_size=%d\n",dp.DisplayImgRow,dp.DisplayImgCol,dp.DisplayImgSize,dp.DisplayCellSize);
  printf("-----------------------------------------------------------------------------------\n\n");
  //===============================================================================================================================================	

	//---------------------------------------------------Create CPU memory to store the output-------------------------------------------------------
	
  checkCuda(cudaMallocHost ((void**)&CPU_InputArray,hp.ImgSize));
  checkCuda(cudaMallocHost ((void**)&CPU_OutputArray,dp.DisplayImgSize));	
  checkCuda(cudaMallocHost ((void**)&CPU_Hist,dp.TotalCells *4*4));	

  checkCuda(cudaMallocHost ((void**)&CPU_FeatureArray,hp.TotalBlocks*sizeof(float)*hp.FeatureSize));	
  
  memcpy(CPU_InputArray,image.data,hp.ImgSize);
 
  checkCuda(launch_helper(GPURuntimes));
  
	printf("-----------------------------------------------------------------\n");
	printf("Tfr CPU->GPU = %5.2f ms ... \nExecution = %5.2f ms ... \nTfr GPU->CPU = %5.2f ms   \n Total=%5.2f ms\n",
			GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
	printf("-----------------------------------------------------------------\n");

  if (!imwrite(argv[2], Mat(dp.DisplayImgRow, dp.DisplayImgCol, CV_8UC1, CPU_OutputArray))) {
		fprintf(stderr, "couldn't write output to disk!\n");
		cudaFreeHost(CPU_OutputArray);
    cudaFreeHost(CPU_InputArray);
	  cudaFreeHost(CPU_FeatureArray);
		exit(EXIT_FAILURE);
	}
 
  WriteNumbers("Feature.txt",CPU_FeatureArray,hp.BlockRow,hp.BlockCol,hp.FeatureSize);
  WriteNumbers("Display_feature.txt",CPU_Hist,dp.CellRow,dp.CellCol,4);
 
  cudaFreeHost(CPU_OutputArray);
  cudaFreeHost(CPU_InputArray);
  cudaFreeHost(CPU_Hist);	
	cudaFreeHost(CPU_FeatureArray);
  printf("\n\n...EXECUTION COMPLETED...\n\n");
  exit(EXIT_SUCCESS);
}

cudaError_t launch_helper(float* Runtimes)
{
	cudaEvent_t time1, time2, time3, time4;

  int   *Orientation;
	float *Gradient;
  uchar *DisplayOrientation;
	uchar *GPU_idata;
	uchar *GPU_odata;
 	//uchar *GPU_displaydata;
  float *GPU_CellHistogram;
  float *GPU_BlockHistogram;
  float *TempDisplayhistogram;
  float *Displayhistogram;
  dim3 threadsPerBlock;
	dim3 numBlocks;
  int i;
  
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
 
 for(i=0;i<2;i++) checkCuda(cudaStreamCreate(&stream[i]));
 
 checkCuda(cudaMalloc((void**)&GPU_idata, hp.ImgSize));
 checkCuda(cudaMalloc((void**)&Gradient, hp.ImgSize*4));
	
 checkCuda(cudaMalloc((void**)&Orientation, hp.ImgSize*4));
 
 
 checkCuda(cudaMalloc((void**)&DisplayOrientation, hp.ImgSize));
 //checkCuda(cudaMalloc((void**)&GPU_displaydata, hp.ImgSize));
 //checkCuda(cudaMallocHost ((void**)&GPU_CellHistogram,hp.TotalCells*sizeof(float)*hp.NumBins));

 checkCuda(cudaMemcpyAsync(GPU_idata, CPU_InputArray, hp.ImgSize, cudaMemcpyHostToDevice,stream[0]));
 
 cudaEventRecord(time2, 0);
 
 threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
 numBlocks = dim3((int)ceil(hp.ImgRow / (float)threadsPerBlock.x), (int)ceil(hp.ImgCol / (float)threadsPerBlock.y));
 Cal_kernel<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_idata,Orientation,Gradient,DisplayOrientation,hp);
 checkCuda(cudaDeviceSynchronize());
 cudaFree(GPU_idata);
 
 //Launch Display Kernel
 
 checkCuda(cudaMalloc((void**)&TempDisplayhistogram, dp.HorzCellsTotal*4*4));
 checkCuda(cudaMalloc((void**)&Displayhistogram, dp.TotalCells *4*4));  
 threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
 numBlocks = dim3((int)ceil(dp.HorzCells / (float)threadsPerBlock.x), (int)ceil(dp.CellCol / (float)threadsPerBlock.y));
 Display_Cell_kernel<<<numBlocks, threadsPerBlock,0,stream[1]>>>(Displayhistogram,TempDisplayhistogram,DisplayOrientation,Gradient,dp);
 
 checkCuda(cudaMallocHost ((void**)&GPU_CellHistogram,hp.TotalCells*sizeof(float)*hp.NumBins));
 threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
 numBlocks = dim3((int)ceil(hp.CellRow / (float)threadsPerBlock.x), (int)ceil(hp.CellCol / (float)threadsPerBlock.y));
 //printf("\n\n...%d %d...\n\n",numBlocks.x,numBlocks.y); 
 Cell_kernel<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_CellHistogram,Orientation,Gradient,hp);
 
 checkCuda(cudaDeviceSynchronize());
 
 checkCuda(cudaMemcpy(CPU_Hist,Displayhistogram , dp.TotalCells *4*4, cudaMemcpyDeviceToHost));
 
  
 cudaFree(TempDisplayhistogram);
 cudaFree(Orientation); cudaFree(Gradient);
 checkCuda(cudaMalloc((void**)&GPU_odata, dp.DisplayImgSize));
 cudaMemset(GPU_odata, 0, dp.DisplayImgSize);
 threadsPerBlock = dim3(4, 4, 4);
 numBlocks = dim3((int)ceil(dp.CellRow / (float)threadsPerBlock.x), (int)ceil(dp.CellCol / (float)threadsPerBlock.y));
 //printf("\n\n...%d %d...\n\n",numBlocks.x,numBlocks.y); 
 display_kernel<<<numBlocks, threadsPerBlock,0,stream[1]>>>(Displayhistogram,GPU_odata,dp);
 
 
 checkCuda(cudaMallocHost ((void**)&GPU_BlockHistogram,hp.TotalBlocks*sizeof(float)*hp.FeatureSize));
 threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
 numBlocks = dim3((int)ceil(hp.BlockRow / (float)threadsPerBlock.x), (int)ceil(hp.BlockCol / (float)threadsPerBlock.y));
 //printf("\n\n...%d %d...\n\n",numBlocks.x,numBlocks.y); 
 Block_kernel<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_BlockHistogram, GPU_CellHistogram, hp);
 
 cudaEventRecord(time3, 0);
 
 checkCuda(cudaMemcpyAsync(CPU_OutputArray, GPU_odata, dp.DisplayImgSize, cudaMemcpyDeviceToHost,stream[1]));
 checkCuda(cudaDeviceSynchronize());
 
 //checkCuda(cudaMemcpy(CPU_CellArray,GPU_CellHistogram , hp.TotalCells*sizeof(float)*hp.NumBins, cudaMemcpyDeviceToHost));
 
 checkCuda(cudaMemcpy(CPU_FeatureArray,GPU_BlockHistogram , hp.TotalBlocks*sizeof(float)*hp.FeatureSize, cudaMemcpyDeviceToHost));
 
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
  for(i=0;i<2;i++) cudaStreamDestroy(stream[i]);
	cudaFree(GPU_odata);
	cudaFree(GPU_idata);
  cudaFree(Orientation);
  cudaFree(Gradient);
  cudaFree(GPU_CellHistogram);
  cudaFree(GPU_BlockHistogram);
	cudaFree(Displayhistogram);
  cudaFree(TempDisplayhistogram);
 	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);
 
  return cudaStatus;
}