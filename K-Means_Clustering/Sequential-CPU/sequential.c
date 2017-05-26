#include "sequential.h"

int main(int argc, char** argv)
{
    int i ,p; 
    
	if (argc != 4){
		printf("Usage: %s <input image> <output image> <Number of clusters> \n", argv[0]);
		exit(EXIT_FAILURE);
	}
	
	TheImage = ReadBMP(argv[1]);
	K = atoi(argv[3]);
  p =0; 
  size = ip.Vpixels*ip.Hpixels;
  
  printf("size: %d pixels\n", size);
	
	//TODO: convert many of these into a single object
	
	//every point has an RGB value
  double*RDataset = (double*)malloc(size*sizeof(double));
  double*GDataset = (double*)malloc(size*sizeof(double));
  double*BDataset = (double*)malloc(size*sizeof(double));
  //every center point has an RGB value
  double*CR  = (double*)malloc(K*sizeof(double));
  double*CG  = (double*)malloc(K*sizeof(double));
  double*CB  = (double*)malloc(K*sizeof(double)); 
  //copy of center points for comparison
  double*OCR = (double*)malloc(K*sizeof(double));
  double*OCG = (double*)malloc(K*sizeof(double));
  double*OCB = (double*)malloc(K*sizeof(double));
  //every point keeps track of its minimum euclidean distance
  double*Euclidean_Dist = (double*)malloc(size*sizeof(double));
  //every point is assigned a cluster label
  int *label = (int*)malloc(size*sizeof(int));
  //every cluster __keeps track of its size?__
  int *NC = (int*)malloc(K*sizeof(int));
  
  //intialize all points to have an infinite value for euclidean distance
  for (i = 0 ; i < size ; i++){
    Euclidean_Dist[i] = UINT_MAX;
  }
  //mark down what we have
  WriteFile(TheImage,RDataset,GDataset, BDataset,Euclidean_Dist,label,"inputData.txt");
  //set the initial center points
  getRandomCentroids(RDataset,GDataset,BDataset,CR,CG,CB,K);
  //calculate the first set of distances
  
  struct timespec start, end;
	double totalTime = 0.0;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	
	euclideanDist(RDataset,GDataset,BDataset,CR,CG,CB,Euclidean_Dist,label);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		totalTime += (double)((	(double)(1000000 * (end.tv_sec - start.tv_sec)) + (double)((end.tv_nsec - start.tv_nsec) / 1000)))/1000;
  //main algorithm loop
  for (i = 0 ; i < 1000 ; i ++){
          //TODO: Shouldn't NC get reset at the beginning of each iteration?
	  //printf("iteration %d\n", i);
	  p = convergence(label,RDataset,GDataset,BDataset,OCR,OCG,OCB,CR,CG,CB,NC,K);
	  //printf("print p :%d",p);
	  if(p == 1){
		  
		  p =0;
		  break;
	  }
	  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	  euclideanDist(RDataset,GDataset,BDataset,CR,CG,CB,Euclidean_Dist,label);
	  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	  totalTime += (double)((	(double)(1000000 * (end.tv_sec - start.tv_sec)) + (double)((end.tv_nsec - start.tv_nsec) / 1000)))/1000;
  }
  WriteFile(TheImage,RDataset,GDataset, BDataset,Euclidean_Dist,label,"outputData.txt");
  unsigned char** OutImage = CreateBlankBMP('0');
  GetImage(OutImage,CR,CG,CB,K,label);
  WriteBMP(OutImage,argv[2]);	
	for(i = 0; i < ip.Vpixels; i++) { free(TheImage[i]); }
    printf("Algorithm execution time: %f ms\n\n",totalTime);
 
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
    //first, copy the original center points into a temp array
    for(j= 0 ; j < K; j++){
      
      OCR[j] = CR[j];
      OCG[j] = CG[j];
      OCB[j] = CB[j];
      CR[j] = 0;
      CG[j] = 0;
      CB[j] = 0;
      //printf("\told centroid #%d:\t%lf %lf %lf\n",j,OCR[j],OCG[j],OCB[j]);
    }
    //loop through every center and add to the sums for its RGB values, each RGB value of every point that was assigned to that center
    //and keep track of the size as it grows
    //TODO: This can be further optimized by looping through each point once and find the center point array index based on label[i]
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
   //TODO: merge with the above loop, if it doesn't get changed
   for(j= 0 ; j < K; j++){
          if(NC[j] == 0){
            CR[j] = OCR[j];
            CG[j] = OCG[j];
            CB[j] = OCB[j];
          }else {
            CR[j] = CR[j]/NC[j];
            CG[j] = CG[j]/NC[j];
            CB[j] = CB[j]/NC[j];}
           //printf("\tnew centroid #%d:\t%lf %lf %lf\n",j,CR[j],CG[j],CB[j]);
    }
   //TODO: instead of keeping count, just set flag = 0 and return when the if condition returns false
       for(j= 0 ; j < K; j++){
        if ((OCR[j] - CR[j])<= 0.001 && (OCG[j] - CG[j])< 0.001 && (OCB[j] - CB[j])< 0.001){
        count += 1; 
        } 
      }
      
      if (count ==  K){
        flag = 1;  
        //printf("count2: %d", count);
    }
  return flag;
}

void euclideanDist(double* RDataset,double* GDataset,double* BDataset,double* CR,double* CG,double* CB,double* Euclidean_Dist,int* label){
 int i , j;
 double temp;
  
 for(i = 0 ; i < size ; i ++){
 //for every point, we are calculating its euclidean distance to every other point and then assigning its label to the closest one.
   for(j = 0 ; j < K; j++){
	   //we skip the square root step because it doesn't affect how the minimum is calculated
	   temp = ((GDataset[i]-CG[j])* (GDataset[i]-CG[j]))+ ((BDataset[i]-CB[j])*(BDataset[i]-CB[j]))+((RDataset[i]-CR[j])*(RDataset[i]-CR[j]));   
	   //printf("%lf\n",temp);
	   if(temp < Euclidean_Dist[i]){ 
		 Euclidean_Dist[i] = temp;
		// printf("%lf\n",Euclidean_Dist[i]);
		 label[i] = j;
	   }
	   //TODO: if they are equal, choose a random one as the new label.
   } 
 } 
}

void getRandomCentroids(double* RDataset,double* GDataset,double* BDataset,double* CR,double* CG,double* CB, int k)
{   
    double scaled ;
    int r ,i;
   srand(time(NULL));
   //TODO: ensure that multiple centers do not get the same random value
    for(i = 0 ; i < k ; i++){
      r = rand() % size;
        CR[i] =  RDataset[r];
        CG[i] =  GDataset[r];
        CB[i] =  BDataset[r];
    }    
}



