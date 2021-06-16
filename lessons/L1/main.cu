#include <iostream>
#include <cstdio>
#include <math.h>


/* Threads --> Execute Instructions
   Warp --> Comprised on multiple threads -- > Executes the threads in lockstep( Not every threads needs to execute all the instructions )instructions
   Thread Blocks --> Group of threads blocks --> 3 Dimensional (x,y,z) --> assigned to shader core
   Grids --> Composed of thread blocks. --> Figure out how a problem is mapped to gpu --> 3 Dimensional
  
   Blocks sizes are in multiple of 32, so threads must be in the set up in the same way
  */

__global__
void add(int n , float *x, float *y){
  int index = threadIdx.x *blockDim.x + threadIdx.x; // index of the thread within the block
  int stride = blockDim.x * gridDim.x; // Number of threads within the block
  for(int i =index; i<n ; i+= stride){
    y[i] = x[i] + y[i];
  }
}


int main(int argc, char* argv[]){
  int N = 1<<20;
  int blocksize = 256;
  int numBlocks = (N - blocksize-1) / blocksize;
  float *x,*y;
  /* Allocated needed memeory to be accessible from CPU or GPU */ 
  cudaMallocManaged(&y, N*sizeof(float));
  cudaMallocManaged(&x, N*sizeof(float));
  // Set Values on the host (CPU)
  for(int i = 0; i < N; i++){
    x[i] = 1.0f;
    y[i] = 2.0f;

  }

  // Run kernel on the gpu
  add<<<numBlocks,blocksize>>>(N,x,y);

  // Waits for the GPU to finish before accessing the host
  cudaDeviceSynchronize();

  float maxError = 0;
  for(int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] -3.0f));
  std::cout << "Max Error: " << maxError << std::endl;
  
  // Free Memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
