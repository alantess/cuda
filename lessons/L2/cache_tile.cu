#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <math.h>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
using std::vector;

/* Dynamic random-access memory (DRAM) is a type of random-access semiconductor memory that stores each bit of data in a memory cell consisting of a tiny capacitor and a transistor. VERY SLOW 
  Cache is easier to access (faster), smaller than the main memeory, and located closer to the processor core. L1 cache is the primary cache..
    - Share Memory (scratchpad) -- User managed L1 cache / Private per threadblock 
    - Cache Tiling: Large Input --> Can't fit all into cache since its smaller than main memory. Puts only pieces of the large inputs into the cache
    - Large input and reducing it at each iteration. Basically working on everything chunks at a time.
    - A & B Matrix ---> A[y][k] * A[k][x]
 */

// 16 x 16 matrix
const int SHMEM_SIZE = 1<<10;
const int N = 1<<10; 

__global__ void tiledMatrixMul(const int *a, const int *b, int *c) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Statically allocated shared memory
  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];

  // Accumulate in temporary variable
  int tmp = 0;

  // Sweep tile across matrix
  for (int i = 0; i < N; i += blockDim.x) {
    // Load in elements for this tile
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] =
        b[i * N + threadIdx.y * N + col];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix
    for (int j = 0; j < blockDim.x; j++) {
      tmp +=
          s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }

  // Write back results
  c[row * N + col] = tmp;
}



// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}


int main(){
  // Matrix 1024x1024
  // Size in bytes
  size_t bytes = N*N *sizeof(int);
  // Allocate to host
  vector<int>h_a(N*N);
  vector<int>h_b(N*N);
  vector<int>h_c(N*N);

  // Initialize matrix
  std::generate(h_a.begin(), h_a.end(), [](){return rand() % 100;});
  std::generate(h_b.begin(), h_b.end(), [](){return rand() % 100;});


  // Device pointers
  int *d_a, *d_b,* d_c;

  //  Allocate memory to gpu
  cudaMalloc(&d_a,bytes);
  cudaMalloc(&d_b,bytes);
  cudaMalloc(&d_c,bytes);



  // Copy memory over to gpu
  cudaMemcpy(d_a,h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,h_b.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c,h_c.data(), bytes, cudaMemcpyHostToDevice);


  int THREADS = 32;
  
  int BLOCKS = (int)ceil(N/THREADS);


  // Use a 3D object
  dim3 blocks(BLOCKS, BLOCKS);
  dim3 threads(THREADS, THREADS);

  /* // Launch kernel */
  tiledMatrixMul<<<blocks, threads>>> (d_a,d_b,d_c );

  // Back to cpu
  cudaMemcpy(h_c.data(),d_c, bytes,cudaMemcpyDeviceToHost);

  // check results

  verify_result(h_a,h_b,h_c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  printf("Completed.");
  return 0;
  }
