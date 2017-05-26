
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

void convertMat2Data (Mat img, double *RData,double *GData,double *BData){
    int i,j,k;
    k=0;
    for(i = 0;i < img.rows ;i++){
        for(j = 0; j < img.cols ;j++){
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
      //printf("old centroids i :%d %lf %lf %lf\n",j,OCR[j],OCG[j],OCB[j]);
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
           //printf("new centroids %lf %lf %lf\n",CR[j],CG[j],CB[j]);
            }
           
      for(j = 0 ; j < K; j++){
        if ((OCR[j] - CR[j])< 0.001 && (OCG[j] - CG[j])< 0.001 && (OCB[j] - CB[j])< 0.001){
        count += 1;
        } 
      }
    //printf("count2*********: %d\n", count);
    if (count ==  K){
        flag = 1;  
        //printf("count2: %d", count);
    }
            
  return flag;
}


