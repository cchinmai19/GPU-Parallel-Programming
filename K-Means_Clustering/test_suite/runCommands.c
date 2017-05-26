#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

int main(int *arg,int** c ){
        char* fileName; int i;
        for(i=0; i<3;i++){
            if(i==0)
              fileName = "../images/color-pencils.bmp";
            else if(i==1)
              fileName = "../images/lenna.bmp";
            else if(i==2)
              fileName = "../images/marbles.bmp";
			
          printf("\nWe are testing image: %s\n",fileName);
          int j;
          printf("\tFirst, sequentially:\n");
          for(j=2; j<17; j++){
             double timeSum = 0;
             char* kmeansCommand = (char*)malloc(100*sizeof(char));
             sprintf(kmeansCommand, "./kmeansSeq %s nonsense.bmp %d",fileName,j);
             int z;
             for(z =0; z<6; ++z){
	             timeSum += system(kmeansCommand)*4;
	             //printf("\ttimeSum is now %f\t",timeSum);
	         }
	         printf("\t\tk = %d\ttime = %d ms\n",j,(int)(timeSum/6000));
             free(kmeansCommand);    
          }
          printf("\n\tThen, in parallel:\n");
          for(j=2; j<17; j++){
             double timeSum = 0;
             char* kmeansCommand = (char*)malloc(100*sizeof(char));
             sprintf(kmeansCommand, "./kmeansPara %s nonsense.bmp %d",fileName,j);
             int z;
             for(z =0; z<6; ++z){
	             timeSum += system(kmeansCommand);
	             //printf("\ttimeSum is now %f\t",timeSum);
	         }
	         printf("\t\tk = %d\ttime = %d ms\n",j,(int)(timeSum/1000));
             free(kmeansCommand);    
          }
          //      
     }

}
