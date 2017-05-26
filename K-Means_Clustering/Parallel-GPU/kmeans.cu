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

//GPU Kernel Stuff
#include "parallel_kernel01.cu"
#include "kmeansUtils.c"

int K ; // number of clusters
int R; // rows 
int C; // cols

//Store all our information here
double *RDataSet,*GDataSet,*BDataSet;
double *Euclidean_distant;
int *label;
double *CR,*CG,*CB ;
double *OCR,*OCG,*OCB;


int main (int argc, char *argv[]){

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
	
 // store the input number of clusters
  	K = atoi(argv[3]);
   
  	Mat image; 
  	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  	if (!image.data){
  		fprintf(stderr,"Could not open or find the image.\n");
  		exit(EXIT_FAILURE);
  	}
  	printf("Loaded in '%s' , size: %d x %d\n", argv[1],image.rows,image.cols);
   
   	R = image.rows;
  	C = image.cols; 
    printf("Size :%d pixels.\n", R*C);
  
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
  			fprintf(stderr,"Failed to create Euclidean_distant using malloc()...\n");
  			exit(EXIT_FAILURE);
  	}
   
   label = (int*)malloc(R*C*sizeof(int));
  		if(label == NULL){
  			fprintf(stderr,"Failed to create label using malloc()...\n");
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
     // better way to keep time:
	double totalTime = 0.0;	
	
//     StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec); 
     
	for (i = 0 ; i < 80 ; i++){
      
     //printf("iteration: %d\n", i);
    
     //launch cuda kernel
      cudaStatus = kernel_launcher(image,RDataSet,GDataSet,BDataSet,CR,CG,CB,Euclidean_distant,label,K,R,C,&totalTime);
      if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "launch_helper failed!\n");
        exit(EXIT_FAILURE);
      } 
      p = convergence(label,RDataSet,GDataSet,BDataSet,OCR,OCG,OCB,CR,CG,CB,NC,K,R,C);
      if(p == 1){
      	break;
      } 
      
      //reset device
      checkCuda(cudaDeviceReset());
    }
    
      //Write_ResultFile((char*)"output.txt",RDataSet,GDataSet,BDataSet,Euclidean_distant,label,R*C);
      string output_filename = argv[2];      
      //printf("\tAttempting to assign new values to the image. . .\n");
      //printf("\tThis is how many labels there are: %d\n",sizeof(label) );
      for(i=0; i < R; i++ ){
        for(j=0; j< C; j++){
        	//printf("\t\tReached index value %d\n",index);
        	//printf("\t\tThis is label[index]: %d\n",label[index]);
        	//printf("\t\tThis is CR[label[index]]: %d\n",CR[label[index]]);
        	//printf("\t\tThis is image.at<Vec3b>(i,j)[0]: %d\n",image.at<Vec3b>(i,j)[0]);
        	
          image.at<Vec3b>(i,j)[0]=(unsigned char) CR[label[index]];
          image.at<Vec3b>(i,j)[1]=(unsigned char) CG[label[index]];
          image.at<Vec3b>(i,j)[2]=(unsigned char) CB[label[index]];
          index = index +1;
          }
        }
    	//printf("\tAttempting to write output file. . .\n");
        if (!imwrite(output_filename, image)) {
          fprintf(stderr, "couldn't write output to disk!\n");
          exit(EXIT_FAILURE);
      }
      
      printf("Saved image: '%s', size = %dx%d\n",output_filename.c_str(), image.rows, image.cols);
      printf("Kernel execution time: %f ms\n",totalTime);
      // Free allocated CPU memory once done
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

