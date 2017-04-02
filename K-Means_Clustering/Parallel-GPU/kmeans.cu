
// kMeans Clustering of images
// Author- Chinmai Panibathe

#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <sys/time.h>
#include <limits.h>
#include <string>
#include <math.h>

// CUDA Libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// OpenCV Libraries 
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;

//other support
cudaError_t launch_helper(Mat image,double *RDataSet,double *GDataSet,double *BDataSet,double *OCR,double *OCG,double *OCB,double *Euclidean_distant,int *label,float* Runtimes,int n,int R, int C);
int convergence(int* label, double* RDataset,double* GDataset,double* BDataset,double* OCR,double* OCG,double* OCB,double* CR,double* CG,double* CB,int *NC, int K,int R,int C);
void Write_ResultFile(char* filename,double *RDataset,double *GDataset,double *BDataset, double *distance,int *label,int len_Data);
void getRandomCentroids(double* RDataset,double* GDataset,double* BDataset,double* CR,double* CG,double* CB,int R,int C, int k);
void convertMat2Data (Mat img, double *RData,double *GData,double *BData);
unsigned char** CreateBlankBMP(unsigned char FILL,int R, int C);
void GetImage(unsigned char** img, int *R, int *G, int *B,int K);
cudaError_t checkCuda(cudaError_t result);

// global variables 
int K ; // number of clusters by the user
int R; // rows 
int C; // cols

//Resultant Arrays
double *RDataSet,*GDataSet,*BDataSet;
double *Euclidean_distant;
int *label;
double *CR,*CG,*CB ;
double *OCR,*OCG,*OCB;

 
__global__ void euclideanDist_kernel(uchar *GPU_i,double *CR,double *CG,double *CB,double *Euclidean_distant,int *label,int r, int c, int n )
{
  int k ;
  double temp;
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row of image
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // column of image
  int idx = j*c*3 + i*3;
  int odx = i + j*c;
  
  for (k = 0 ; k < n ; k ++) {      
      temp = ((double)GPU_i[idx] - CR[k]) * ((double)GPU_i[idx]- CR[k]) + ((double)GPU_i[idx+1] - CG[k]) * 
                               ((double)GPU_i[idx+1]- CG[k]) +((double)GPU_i[idx+2] - CB[k]) * ((double)GPU_i[idx+2]- CB[k]);                            
      if(temp < Euclidean_distant[odx]) {
          Euclidean_distant[odx] = temp;
          label[odx] = k; 
      }                             
      } 
}

int main (int argc, char *argv[]){

  	float GPURuntimes[4];
    struct            timeval 	t;
    double         		StartTime, EndTime;
    double         		TimeElapsed;
  	cudaError_t cudaStatus;
    int i,K,R,C,p,j;
    double *RDataSet,*GDataSet,*BDataSet;
    double *Euclidean_distant;
    int *label;
    int index;
    double *CR,*CG,*CB ;
  
    p = 0;
 	  index = 0;
      
      
  	if (argc != 4){
  		printf("Usage: %s <input image> <output image> <Number of clusters> \n", argv[0]);
  		exit(EXIT_FAILURE);
  	}
	
 // define number of clusters
  	K = atoi(argv[3]);
   
   //Store input image1 and image2 in Mat1 and Mat2
  	Mat image; 
  	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  	if (!image.data){
  		fprintf(stderr,"Could not open or find the image.\n");
  		exit(EXIT_FAILURE);
  	}
  	printf("Loaded First image %s , size: %d x %d\n", argv[1],image.rows,image.cols);
   
   	R = image.rows;
  	C = image.cols; 
    //printf("rows :%d col: %d\n", R, C);
  
 //Write Image Data to a file 
 
   RDataSet =  (double*)malloc(R*C*sizeof(double));
   GDataSet =  (double*)malloc(R*C*sizeof(double));
   BDataSet =  (double*)malloc(R*C*sizeof(double));
   int *NC = (int*)malloc(K*sizeof(int));
   CR = (double*)malloc(K*sizeof(double));
   CG = (double*)malloc(K*sizeof(double));
   CB = (double*)malloc(K*sizeof(double));
   OCR = (double*)malloc(K*sizeof(double));
   OCG = (double*)malloc(K*sizeof(double));
   OCB = (double*)malloc(K*sizeof(double));

   convertMat2Data(image,RDataSet,GDataSet,BDataSet);
  	
    Euclidean_distant = (double*)malloc(R*C*sizeof(double));
  		if(Euclidean_distant == NULL){
  			fprintf(stderr,"OOPS. can't create Euclidean_distant using malloc()...\n");
  			exit(EXIT_FAILURE);
  	}
   
   label = (int*)malloc(R*C*sizeof(int));
  		if(label == NULL){
  			fprintf(stderr,"OOPS. can't create label using malloc()...\n");
  			exit(EXIT_FAILURE);
  	}

     for(i=0; i<(R*C); i++){
       Euclidean_distant[i] = UINT_MAX;
     } 
     
     getRandomCentroids(RDataSet,GDataSet,BDataSet,CR,CG,CB,R,C,K);
       
     for(i = 0 ; i < K; i ++){
         OCR[i] = 0;
         OCG[i] = 0 ;
         OCB[i] = 0;
     }
     
     gettimeofday(&t, NULL);
     StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec); 
     
for (i = 0 ; i < 100 ; i++){
      
     printf("iteration: %d\n", i);
    
     //launch cuda helper
    	cudaStatus = launch_helper(image,RDataSet,GDataSet,BDataSet,CR,CG,CB,Euclidean_distant,label,GPURuntimes,K,R,C);
              if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "launch_helper failed!\n");
                exit(EXIT_FAILURE);
              } 
             
      printf("-----------------------------------------------------------------\n");
      printf("Tfr CPU->GPU = %5.2f ms ... \nExecution = %5.2f ms ... \nTfr GPU->CPU = %5.2f ms   \n Total=%5.2f ms\n",
      GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
      printf("-----------------------------------------------------------------\n");         
   
      p = convergence(label,RDataSet,GDataSet,BDataSet,OCR,OCG,OCB,CR,CG,CB,NC,K,R,C);
      if(p == 1){
      break;} 
      
      //reset device
      checkCuda(cudaDeviceReset());
      }
      gettimeofday(&t, NULL);
	    EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	    TimeElapsed=(EndTime-StartTime)/1000.00; 
      Write_ResultFile("Resultant_Data.txt",RDataSet,GDataSet,BDataSet,Euclidean_distant,label,R*C);
      string output_filename = argv[2];      
      Mat Cluster(R,C, CV_64F);
      
      for(i=0; i < R; i++ ){
        for(j=0; j< C; j++){
          Cluster.at<double>(i,j) = label[index] * (255/K);
          index = index +1;
          }
        }
    
        if (!imwrite(output_filename, Cluster)) {
          fprintf(stderr, "couldn't write output to disk!\n");
          exit(EXIT_FAILURE);
      }
      
      printf("Saved image '%s', size = %dx%d (dims = %d).\n",output_filename.c_str(), Cluster.rows, Cluster.cols, Cluster.dims);
      // Free allocated CPU memory once done
      printf("\n\nTotal execution time: %9.4f ms",TimeElapsed);
      free(Euclidean_distant);
      free(label);
      free(RDataSet);
      free(GDataSet);
      free(BDataSet);
      free(CR);
      free(CB);
      free(CG);
      free(OCR);
      free(OCB);
      free(OCG);
    	exit(EXIT_SUCCESS);	
}

cudaError_t launch_helper(Mat image,double *RDataSet,double *GDataSet,double *BDataSet,double *CR,double *CG,double *CB,double *Euclidean_distant,int *label,float* Runtimes,int n,int R, int C)
{    
	//Declare variables
	cudaEvent_t time1, time2, time3, time4;
	int TotalGPUSize;
	uchar *GPU_idata;
	double *GPU_CR;
	double *GPU_CG;
	double *GPU_CB;
	double *GPU_EuclDist;
	int *GPU_label;
	int *GPU_NC;
	dim3 threadsPerBlock;
	dim3 numBlocks;
	cudaError_t cudaStatus;

	// use the first GPU (not necessarily the fastest)
	cudaStatus = cudaSetDevice(0);  
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
	//goto Error;
	  }

	//Event declaration 
	cudaEventCreate(&time1);
	cudaEventCreate(&time2);
	cudaEventCreate(&time3);
	cudaEventCreate(&time4);

	//Allocate memory in GPU
	cudaEventRecord(time1, 0);
	TotalGPUSize = 3*R*C * sizeof(uchar); 

	checkCuda(cudaMalloc((void**)&GPU_idata, TotalGPUSize));
	checkCuda(cudaMalloc((void**)&GPU_CR, n*sizeof(double)));
	checkCuda(cudaMalloc((void**)&GPU_CG, n*sizeof(double)));
	checkCuda(cudaMalloc((void**)&GPU_CB, n*sizeof(double)));
	checkCuda(cudaMalloc((void**)&GPU_NC, n*sizeof(int)));
	checkCuda(cudaMalloc((void**)&GPU_EuclDist, R*C*sizeof(double)));   
	checkCuda(cudaMalloc((void**)&GPU_label, R*C*sizeof(int))); 
    
	//Transfer data from CPU to GPU 

	cudaEventRecord(time2, 0);
	checkCuda(cudaMemcpy(GPU_idata, image.data, TotalGPUSize, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(GPU_EuclDist, Euclidean_distant, R*C*sizeof(double), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(GPU_CR, CR, n*sizeof(double), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(GPU_CG, CG, n*sizeof(double), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(GPU_CB, CB, n*sizeof(double), cudaMemcpyHostToDevice));
      
  	// Launch a kernel on the GPU with one thread for each pixel.
   
	threadsPerBlock = dim3(32,32);
	printf("x: %d y:%d\n", threadsPerBlock.x,threadsPerBlock.y);
	numBlocks = dim3(C/threadsPerBlock.x,R/threadsPerBlock.y);

	euclideanDist_kernel<<<numBlocks, threadsPerBlock,0>>>(GPU_idata,GPU_CR,GPU_CG,GPU_CB,GPU_EuclDist,GPU_label,R,C,n);	
	checkCuda(cudaGetLastError());
		checkCuda(cudaDeviceSynchronize());  	

	  //Copy output (results) from GPU buffer to host (CPU) memory.

	cudaEventRecord(time3, 0);
	checkCuda(cudaMemcpy(Euclidean_distant, GPU_EuclDist, R*C*sizeof(double), cudaMemcpyDeviceToHost));
	cudaStatus = (cudaMemcpy(label, GPU_label, R*C*sizeof(int), cudaMemcpyDeviceToHost));
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudamemcpy LABEL failed!\n");
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
  	cudaFree(GPU_idata);
	cudaFree(GPU_CR);
	cudaFree(GPU_CG);
	cudaFree(GPU_CB);
	cudaFree(GPU_EuclDist);
	cudaFree(GPU_label);
	cudaFree(GPU_NC);
  	cudaEventDestroy(time1);
  	cudaEventDestroy(time2);
  	cudaEventDestroy(time3);
  	cudaEventDestroy(time4);
  
  	return cudaStatus;
}

void getRandomCentroids(double* RDataset,double* GDataset,double* BDataset,double* CR,double* CG,double* CB, int R, int C,int k)
{   
	int r ,i;  
	srand(time(NULL));
	for(i = 0 ; i < k ; i++){
		r = rand() % R*C;
		CR[i] =  RDataset[r];
		CG[i] =  GDataset[r];
		CB[i] =  BDataset[r];
	}    
}

void Write_ResultFile(char* filename,double *RDataset,double *GDataset,double *BDataset,double *distance, int *label,int len_Data)
{
	int i;
	FILE* f = fopen(filename, "w");
	if(f == NULL)
	{
		printf("\n\n%s NOT FOUND\n\n",filename);
		exit(1);
	}
	
	for (i=0;i<len_Data;i++)
	{
		fprintf(f, "%lf %lf %lf %lf %d", RDataset[i],GDataset[i],BDataset[i],distance[i],label[i]);
		fprintf(f,"\n");
	}
}

cudaError_t checkCuda(cudaError_t result)
{   
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s %d\n", cudaGetErrorString(result));
    // We should be free()ing CPU+GPU memory here, but we're relying on the OS
    // to do it for us.
  }
  return result;
}

void convertMat2Data (Mat img, double *RData,double *GData,double *BData){
    int i,j,k;
    k=0;
    for(i = 0;i < img.rows ;i++){
        for(j = 0; j < img.cols ;j = j++){
            RData[k]= img.at<Vec3b>(i, j)[0];
            GData[k]= img.at<Vec3b>(i, j)[1];
            BData[k]= img.at<Vec3b>(i, j)[2];
            k = k+1;
        }
    }
}

int convergence(int* label, double* RDataset,double* GDataset,double* BDataset,double* OCR,double* OCG,double* OCB,double* CR,double* CG,double* CB,int *NC, int K,int R, int C){

    int i,j,count,flag; 
    count = 0;
    flag = 0;
    
    for(j= 0 ; j < K; j++){
      
      OCR[j] = CR[j];
      OCG[j] = CG[j];
      OCB[j] = CB[j];
      CR[j] = 0;
      CG[j] = 0;
      CB[j] = 0;
      printf("old centroids i :%d %lf %lf %lf\n",j,OCR[j],OCG[j],OCB[j]);
    }
    
    for(j= 0 ; j < K; j++){
      count = 0;
      for(i = 0 ; i < R*C; i++){ 
          if (label[i] == j){
            NC[j] = count++;
            CR[j] += RDataset[i];
            CG[j] += GDataset[i];
            CB[j] += BDataset[i];
          }else {
          NC[j] = count;
          }
      }
    } 
   count = 0 ; 
   
   for(j = 0 ; j < K; j++){
          if(NC[j] == 0){
            CR[j] = OCR[j];
            CG[j] = OCG[j];
            CB[j] = OCB[j];
          }else {
            CR[j] = CR[j]/NC[j];
            CG[j] = CG[j]/NC[j];
            CB[j] = CB[j]/NC[j];}
           printf("new centroids %lf %lf %lf\n",CR[j],CG[j],CB[j]);
            }
           
      for(j = 0 ; j < K; j++){
        if ((OCR[j] - CR[j])< 0.001 && (OCG[j] - CG[j])< 0.001 && (OCB[j] - CB[j])< 0.001){
        count += 1;
        } 
      }
    //printf("count2*********: %d\n", count);
    if (count ==  K){
        flag = 1;  
        printf("count2: %d", count);
    }
            
  return flag;
}


