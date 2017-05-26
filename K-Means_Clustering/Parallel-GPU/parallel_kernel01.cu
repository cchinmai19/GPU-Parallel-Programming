cudaError_t checkCuda(cudaError_t result)
{   
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s %d\n", cudaGetErrorString(result));
  }
  return result;
}

__global__ void euclideanDist_kernel(uchar *GPU_i,double *CR,double *CG,double *CB,double *Euclidean_distant,int *label,int r, int c, int n )
{
  int k ;
  double temp;
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // row of image
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // column of image
  int idx = j*c*3 + i*3;
  int odx = i + j*c;
  
  for (k = 0 ; k < n ; k ++) {      
      temp = ((double)GPU_i[idx] - CR[k]) * ((double)GPU_i[idx]- CR[k]) + ((double)GPU_i[idx+1] - CG[k]) * 
                               ((double)GPU_i[idx+1]- CG[k]) +((double)GPU_i[idx+2] - CB[k]) * ((double)GPU_i[idx+2]- CB[k]);                            
      if(temp < Euclidean_distant[odx]) {
          Euclidean_distant[odx] = temp;
          label[odx] = k; 
      }                             
   }
   //if(label[odx] > k)
   	//label[odx] = k/2;
}


cudaError_t kernel_launcher(Mat image,double *RDataSet,double *GDataSet,double *BDataSet,double *CR,double *CG,double *CB,double *Euclidean_distant,int *label,int n,int R, int C, double* fullTime)
{    
	int TotalGPUSize;
	uchar *GPU_idata;
	double *GPU_CR;
	double *GPU_CG;
	double *GPU_CB;
	double *GPU_EuclDist;
	int *GPU_label;
	int *GPU_NC;
	dim3 threadsPerBlock;
	dim3 numBlocks;
	cudaError_t cudaStatus;
	
	cudaStatus = cudaSetDevice(0);  
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaSetDevice failed!\n");
	  }

	TotalGPUSize = 3*R*C * sizeof(uchar); 

	cudaMalloc((void**)&GPU_idata, TotalGPUSize);
	cudaMalloc((void**)&GPU_CR, n*sizeof(double));
	cudaMalloc((void**)&GPU_CG, n*sizeof(double));
	cudaMalloc((void**)&GPU_CB, n*sizeof(double));
	cudaMalloc((void**)&GPU_NC, n*sizeof(int));
	cudaMalloc((void**)&GPU_EuclDist, R*C*sizeof(double)); 
	cudaMalloc((void**)&GPU_label, R*C*sizeof(int));
    
	cudaMemcpy(GPU_idata, image.data, TotalGPUSize, cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_EuclDist, Euclidean_distant, R*C*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_CR, CR, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_CG, CG, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_CB, CB, n*sizeof(double), cudaMemcpyHostToDevice);
      
  	// Launch a kernel on the GPU with one thread for each pixel.
   
	threadsPerBlock = dim3(4,256);
	//printf("x: %d y:%d\n", threadsPerBlock.x,threadsPerBlock.y);
	numBlocks = dim3(C/threadsPerBlock.x,R/threadsPerBlock.y);
	
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	euclideanDist_kernel<<<numBlocks, threadsPerBlock>>>(GPU_idata,GPU_CR,GPU_CG,GPU_CB,GPU_EuclDist,GPU_label,R,C,n);
	cudaGetLastError();
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	*fullTime += (double)((	(double)(1000000 * (end.tv_sec - start.tv_sec)) + (double)((end.tv_nsec - start.tv_nsec) / 1000)))/1000;

	cudaMemcpy(Euclidean_distant, GPU_EuclDist, R*C*sizeof(double), cudaMemcpyDeviceToHost);
	cudaStatus = (cudaMemcpy(label, GPU_label, R*C*sizeof(int), cudaMemcpyDeviceToHost));
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudamemcpy LABEL failed!\n");
	}

  	cudaFree(GPU_idata);
	cudaFree(GPU_CR);
	cudaFree(GPU_CG);
	cudaFree(GPU_CB);
	cudaFree(GPU_EuclDist);
	cudaFree(GPU_label);
	cudaFree(GPU_NC);
  
  	return cudaStatus;
}

