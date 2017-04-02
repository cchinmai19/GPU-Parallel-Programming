//ECE 406 final project
// Sarang Lele, Karthik Dinesh, Chinmai
//Cuda support programs

struct HogProp
{
	int ImgRow,ImgCol;
	int CellRow,CellCol;
  int TotalCells;
  int BlockRow,BlockCol;
  int TotalBlocks;
  int FeatureSize;
  int ImgSize;
	int CellSize;
	int BlockSize;
	int BlockOverlap;
	int NumBins;
	int Orientation;
};

struct DisplayProp
{
	int ImgRow,ImgCol;
	int CellRow,CellCol;
  int TotalCells;
  int HorzCells;
  int HorzCellsTotal;
  //int BlockRow,BlockCol;
  //int TotalBlocks;
  //int FeatureSize;
  int ImgSize;
	int CellSize;
	//int BlockSize;
	//int BlockOverlap;
	int NumBins;
  int DisplayCellSize;
  int DisplayImgRow,DisplayImgCol,DisplayImgSize;
	//int Orientation;
};

#define PI 3.141592654
//extern struct HogProp hp;

cudaError_t checkCuda(cudaError_t result);
int checkerror(int argc, char *argv[], struct HogProp hp);
void WriteNumbers(char* filename, float *features, int row, int col, int numbins);