struct ImgProp
{
	int Hpixels;
	int Vpixels;
	unsigned char HeaderInfo[54];
	unsigned long int Hbytes;
};

unsigned char** CreateOutputBlankBMP(unsigned char FILL);
unsigned char** CreateBlankBMP(unsigned char FILL);
unsigned char** ReadBMP(char* );
void WriteBMP(unsigned char** , char*);
void WriteShrunkBMP(unsigned char** , char*);

extern struct ImgProp ip;
extern int x,y,NewRowBytes;
