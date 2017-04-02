#include <pthread.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "ImageStuff.h"

#define REPS 	     1
#define MAXTHREADS   128

int 			x,y,NewRowBytes;
int 			shrinkRatiox;
int				shrinkRatioy;
long  			NumThreads;         		// Total number of threads working in parallel
int 	     	ThParam[MAXTHREADS];		// Thread parameters ...
pthread_t      	ThHandle[MAXTHREADS];		// Thread handles
pthread_attr_t 	ThAttr;						// Pthread attrributes
void* (*MTShrinkFunc)(void *arg);			// Function pointer to flip the image, multi-threaded version
struct ImgProp 	ip;
unsigned char**	TheImage;					// This is the main image
unsigned char** ShrunkImage;

void *MTShrink(void* tid)
{
    int row, col;

    long ts = *((int *) tid);       	// My thread ID is stored here
    ts *= ip.Vpixels/NumThreads;			// start index
	long te = ts+ip.Vpixels/NumThreads-1; 	// end index
	int var = shrinkRatiox*3;
	for(row=ts; row<=te; row= row+shrinkRatioy)
		{
			col=0;
			while(col<ip.Hbytes)
			{
				ShrunkImage[(int)ceil(row/shrinkRatioy)][(int)ceil(col/shrinkRatiox)]   = TheImage[row][col];
				ShrunkImage[(int)ceil(row/shrinkRatioy)][(int)ceil(col/shrinkRatiox+1)] = TheImage[row][col+1];
				ShrunkImage[(int)ceil(row/shrinkRatioy)][(int)ceil(col/shrinkRatiox+2)] = TheImage[row][col+2];
				
				col = col + var ;
			}
		}
    pthread_exit(0);
}


void *MTRShrink(void* tid)
{
    int row, col;

    long ts = *((int *) tid);       	// My thread ID is stored here
    ts *= y/NumThreads;			// start index
	long te = ts+y/NumThreads-1; 
	for(row=ts; row<=te; row++)
		{
			
			col=0;
			while(col < NewRowBytes)
			{
				ShrunkImage[row][col]   = TheImage[(int)(row*shrinkRatioy)][col*shrinkRatiox];
				ShrunkImage[row][col+1] = TheImage[(int)(row*shrinkRatioy)][col*shrinkRatiox+1];
				ShrunkImage[row][col+2] = TheImage[(int)(row*shrinkRatioy)][col*shrinkRatiox+2];
				col = col + 3;
			}
		}
    pthread_exit(0);
}

int main(int argc, char** argv)
{
	
    int 				a,i,ThErr;
    struct timeval 		t;
    double         		StartTime, EndTime;
    double         		TimeElapsed;
	
	
	
    switch (argc){
		case 5 : NumThreads=8;  shrinkRatiox = atoi(argv[3]);	shrinkRatioy =atoi(argv[4]);;	break;
		default: printf("\n\nUsage: imshrunk input output [thread count] Xratio Yratio");
				 printf("\n\nExample: imshrunk infilename.bmp outname.bmp 8 3 2\n\n");
				 return 0;
    }
	if((isdigit(shrinkRatiox)) && (isdigit(shrinkRatioy))) {
		printf("Please enter digits '%d' and '%d' is invalid.... Exiting abruptly ...\n",shrinkRatiox,shrinkRatioy);
		exit(EXIT_FAILURE);
	}
	
	if((NumThreads<=1) || (NumThreads>MAXTHREADS)){
            printf("\nNumber of threads must be between 1 and %u... Exiting abruptly\n",MAXTHREADS);
            exit(EXIT_FAILURE);
	}else{ 	if(NumThreads != 1){
				printf("\nExecuting the multi-threaded version with %u threads to shrink image to %d : %d ratio...\n",NumThreads,shrinkRatiox,shrinkRatioy );
				MTShrinkFunc = MTRShrink;
			}
	}
	
	TheImage = ReadBMP(argv[1]);
	
	x = ceil(ip.Hpixels/shrinkRatiox);
	y = ceil(ip.Vpixels/shrinkRatioy);
	NewRowBytes = (x*3 + 3) & (~3);
	
	ShrunkImage = CreateOutputBlankBMP(255);

	gettimeofday(&t, NULL);
    StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	
    if(NumThreads >1){
		pthread_attr_init(&ThAttr);
		pthread_attr_setdetachstate(&ThAttr, PTHREAD_CREATE_JOINABLE);
		for(a=0; a<REPS; a++){
			for(i=0; i<NumThreads; i++){
				ThParam[i] = i;
				ThErr = pthread_create(&ThHandle[i], &ThAttr,MTShrinkFunc, (void *)&ThParam[i]);
				if(ThErr != 0){
					printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
					exit(EXIT_FAILURE);
				}
			}
			for(i=0; i<NumThreads; i++){
				pthread_join(ThHandle[i], NULL);
			}
		}
	}
	
    gettimeofday(&t, NULL);
    EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	TimeElapsed=(EndTime-StartTime)/1000.00;
	TimeElapsed/=(double)REPS;
	
    //merge with header and write to file
    WriteShrunkBMP(ShrunkImage, argv[2]);
    
    printf("\n\nTotal execution time: %9.4f ms ",TimeElapsed);
    printf(" (%6.3f ns/pixel)\n", 1000000*TimeElapsed/(double)(ip.Hpixels*ip.Vpixels));
    
    return (EXIT_SUCCESS);
}
