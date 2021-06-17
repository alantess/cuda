#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::vector;

__global__ void matrixMul(const int *a, const int *b, int *c, int N) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Iterate over row, and down column
  c[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    // Accumulate results for a single element
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  }
}


// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
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
  int n  = 1<<10;
  // Size in bytes
  size_t bytes = n*n *sizeof(int);
  // Allocate to host
  vector<int>h_a(n*n);
  vector<int>h_b(n*n);
  vector<int>h_c(n*n);

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
  
  int BLOCKS = n / THREADS;


  // Use a 3D object
  dim3 blocks(BLOCKS, BLOCKS);
  dim3 threads(THREADS, THREADS);

  /* // Launch kernel */
  matrixMul<<<blocks, threads>>> (d_a,d_b,d_c, n);

  // Back to cpu
  cudaMemcpy(h_c.data(),d_c, bytes,cudaMemcpyDeviceToHost);

  // check results

  verify_result(h_a,h_b,h_c,n);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  printf("Completed.");
  return 0;
 }
