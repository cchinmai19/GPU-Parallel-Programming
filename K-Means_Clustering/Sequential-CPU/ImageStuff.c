#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "ImageStuff.h"

struct ImgProp  ip;

unsigned char** CreateBlankBMP(unsigned char FILL)
{
    int i,j;

	unsigned char** img = (unsigned char **)malloc(ip.Vpixels * sizeof(unsigned char*));
    for(i=0; i<ip.Vpixels; i++){
        img[i] = (unsigned char *)malloc(ip.Hbytes * sizeof(unsigned char));
		memset((void *)img[i],FILL,(size_t)ip.Hbytes); // zero out every pixel
    }
    return img;
}

unsigned char** ReadBMP(char* filename)
{
	int i;
	FILE* f = fopen(filename, "rb");
	if(f == NULL){
		printf("\n\n%s NOT FOUND\n\n",filename);
		exit(1);
	}

	unsigned char HeaderInfo[54];
	fread(HeaderInfo, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&HeaderInfo[18];
	int height = *(int*)&HeaderInfo[22];

	//copy header for re-use
	for(i=0; i<54; i++) {
		ip.HeaderInfo[i] = HeaderInfo[i];
	}

	ip.Vpixels = height;
	ip.Hpixels = width;
	int RowBytes = (width*3 + 3) & (~3);
	ip.Hbytes = RowBytes;

	printf("\n   Input BMP File name: %20s  (%u x %u)\n",filename,ip.Hpixels,ip.Vpixels);

	unsigned char tmp;
	unsigned char **TheImage = (unsigned char **)malloc(height * sizeof(unsigned char*));
	for(i=0; i<height; i++) {
		TheImage[i] = (unsigned char *)malloc(RowBytes * sizeof(unsigned char));
	}

	for(i = 0; i < height; i++) {
		fread(TheImage[i], sizeof(unsigned char), RowBytes, f);
	}
	fclose(f);
	return TheImage;  // remember to free() it in caller!
}


void WriteBMP(unsigned char** img, char* filename)
{
	FILE* f = fopen(filename, "wb");
	if(f == NULL){
		printf("\n\nFILE CREATION ERROR: %s\n\n",filename);
		exit(1);
	}

	unsigned long int x,y;
	unsigned char temp;

	//write header
	for(x=0; x<54; x++) {	fputc(ip.HeaderInfo[x],f);	}

	//write data
	for(x=0; x<ip.Vpixels; x++) {
		for(y=0; y<ip.Hbytes; y++){
			temp=img[x][y];
			fputc(temp,f);
		}
	}
	printf("\n  Output BMP File name: %20s  (%u x %u)\n",filename,ip.Hpixels,ip.Vpixels);
	fclose(f);
}



void GetImage(unsigned char** img, int *R, int *G, int *B,int K)
{
    int index = 0,i,j,l;
    l = 255/K;
    
    for(i = 0; i < ip.Vpixels;++i)
    {
        for(j = 0; j < ip.Hbytes;j +=3)
        {
            img[i][j] = (unsigned char)l*R[index];
            img[i][j+1] = (unsigned char)l*G[index];
            img[i][j+2] = (unsigned char)l*B[index];
            index++;
        }
    }
}

void WriteFile(unsigned char** img,double *RDataset,double *GDataset,double *BDataset,double *EuclDataset,int *label,char* filename)
{
	FILE* f = fopen(filename, "w");
	if(f == NULL){
		printf("\n\nFILE CREATION ERROR: %s\n\n",filename);
		exit(1);
	}
	unsigned long int x,y,i;
  i = 0;
	
	for(x=0; x<ip.Vpixels; x++) {
		for(y=0; y<ip.Hbytes; y += 3){
     RDataset[i] = (double)img[x][y];
     GDataset[i] = (double)img[x][y+1];
     BDataset[i] = (double)img[x][y+2];
     fprintf(f, "%u %u %u %lf %d", img[x][y],img[x][y+1],img[x][y+2],EuclDataset[i],label[i]);
     fprintf(f,"\n");
     i += 1;
		}
	}
	fclose(f);
}
