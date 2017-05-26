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


