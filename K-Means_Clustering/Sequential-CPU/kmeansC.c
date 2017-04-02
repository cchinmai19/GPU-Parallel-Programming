#include <pthread.h>
#include <math.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include "ImageStuff.h"

void getRandomCentroids(double* RDataset,double* GDataset,double* BDataset,double* CR,double* CG,double* CB, int k);
void euclideanDist(double* RDataset,double* GDataset,double* BDataset,double* CR,double* CG,double* CB,double* Euclidean_Dist,int* label);
int convergence(int* label, double* RDataset,double* GDataset,double* BDataset,double* OCR,double* OCG,double* OCB,double* CR,double* CG,double* CB,int *NC, int K);
unsigned char**	TheImage;
unsigned char**	Label_Image;
struct ImgProp 	ip;
int size;
int K ; // Number of Clusters

int main(int argc, char** argv)
{	
	  struct            timeval 	t;
    double         		StartTime, EndTime;
    double         		TimeElapsed;
    int               i ,p; 
    
	if (argc != 4){
		printf("Usage: %s <input image> <output image> <Number of clusters> \n", argv[0]);
		exit(EXIT_FAILURE);
	}
	
	TheImage = ReadBMP(argv[1]);
	K = atoi(argv[3]);
  p =0; 
  size = ip.Vpixels*ip.Hpixels;
  
  printf("size: %d\n", size);
	
  double*RDataset = (double*)malloc(size*sizeof(double));
  double*GDataset = (double*)malloc(size*sizeof(double));
  double*BDataset = (double*)malloc(size*sizeof(double));
  double*CR  = (double*)malloc(K*sizeof(double));
  double*CG  = (double*)malloc(K*sizeof(double));
  double*CB  = (double*)malloc(K*sizeof(double)); 
  double*OCR = (double*)malloc(K*sizeof(double));
  double*OCG = (double*)malloc(K*sizeof(double));
  double*OCB = (double*)malloc(K*sizeof(double));
  double*Euclidean_Dist = (double*)malloc(size*sizeof(double));
  int *label = (int*)malloc(size*sizeof(int));
  int *NC = (int*)malloc(K*sizeof(int));
  
  for (i = 0 ; i < size ; i++){
    Euclidean_Dist[i] = UINT_MAX;
  }
  
  WriteFile(TheImage,RDataset,GDataset, BDataset,Euclidean_Dist,label,"data.txt");
  getRandomCentroids(RDataset,GDataset,BDataset,CR,CG,CB,K);
  euclideanDist(RDataset,GDataset,BDataset,CR,CG,CB,Euclidean_Dist,label);
  
	gettimeofday(&t, NULL);
  StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
  for (i = 0 ; i < 1000 ; i ++){
  printf("iteration %d\n", i);
  p = convergence(label,RDataset,GDataset,BDataset,OCR,OCG,OCB,CR,CG,CB,NC,K);
  printf("print p :%d",p);
  if(p == 1){
  p =0;
  break;}
  euclideanDist(RDataset,GDataset,BDataset,CR,CG,CB,Euclidean_Dist,label);
  }
  gettimeofday(&t, NULL);
	EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	TimeElapsed=(EndTime-StartTime)/1000.00;
  WriteFile(TheImage,RDataset,GDataset, BDataset,Euclidean_Dist,label,"data.txt");
	//WriteBMP(TheImage, argv[2]);
  unsigned char** OutImage = CreateBlankBMP('0');
  GetImage(OutImage,label,label,label,K);
  WriteBMP(OutImage,argv[2]);	
	for(i = 0; i < ip.Vpixels; i++) { free(TheImage[i]); }
  printf("\n\nTotal execution time: %9.4f ms",TimeElapsed);
 
  free(TheImage);
  free(RDataset);
  free(GDataset);
  free(BDataset);
  free(Euclidean_Dist);
  free(CR);
	free(CG);
  free(CB);
  free(OCR);
	free(OCG);
  free(OCB);
  free(NC);
  free(label);
  
	return (EXIT_SUCCESS);
}

int convergence(int* label, double* RDataset,double* GDataset,double* BDataset,double* OCR,double* OCG,double* OCB,double* CR,double* CG,double* CB,int *NC, int K){

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
      for(i = 0 ; i < size; i++){ 
          if (label[i] == j){
            NC[j] = count++;
            CR[j] += RDataset[i];
            CG[j] += GDataset[i];
            CB[j] += BDataset[i];
              //printf("j:%d count: %d\n",j,NC[j]);
          }else {
          NC[j] = count;
          }
      }
    }
   count = 0 ; 
   
   for(j= 0 ; j < K; j++){
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
   
       for(j= 0 ; j < K; j++){
        if ((OCR[j] - CR[j])<= 0.001 && (OCG[j] - CG[j])< 0.001 && (OCB[j] - CB[j])< 0.001){
        count += 1; 
        } 
      }
      
      if (count ==  K){
        flag = 1;  
        printf("count2: %d", count);
    }
  return flag;
}

void euclideanDist(double* RDataset,double* GDataset,double* BDataset,double* CR,double* CG,double* CB,double* Euclidean_Dist,int* label){
 int i , j;
 double temp;
  
 for(i = 0 ; i < size ; i ++){
   for(j = 0 ; j < K; j++){
   
   temp = ((GDataset[i]-CG[j])* (GDataset[i]-CG[j]))+ ((BDataset[i]-CB[j])*(BDataset[i]-CB[j]))+((RDataset[i]-CR[j])*(RDataset[i]-CR[j]));   
   //printf("%lf\n",temp);
   if(temp < Euclidean_Dist[i]){ 
     Euclidean_Dist[i] = temp;
    // printf("%lf\n",Euclidean_Dist[i]);
     label[i] = j; 
   }
   } 
 } 
}

void getRandomCentroids(double* RDataset,double* GDataset,double* BDataset,double* CR,double* CG,double* CB, int k)
{   
    double scaled ;
    int r ,i;  
   srand(time(NULL));
    for(i = 0 ; i < k ; i++){
      r = rand() % size;
        CR[i] =  RDataset[r];
        CG[i] =  GDataset[r];
        CB[i] =  BDataset[r];
    }    
}



