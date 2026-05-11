#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init(int *bucket){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  bucket[i] = 0;
}
__global__ void reduction(int *bucket, int *key){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(&bucket[key[i]], 1);
}
__global__ void scan(int *bucket, int *prefix, int *tmp, int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  prefix[i]=bucket[i];
  __syncthreads();

  for(int j=1;j<range;j<<=1){
    tmp[i]=prefix[i];
    __syncthreads();
    if(i>=j)prefix[i]+=tmp[i-j];
    __syncthreads();
  }
}
__global__ void writeback(int *key, int *prefix){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int start = (i==0)? 0:prefix[i-1];
  int end = prefix[i]-1;
  for(int j=start;j<=end;j++)key[j]=i;
}

int main() {
  int n = 50;
  int range = 5;
  int *key;
  int *bucket;
  int *prefix;
  int *tmp;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&prefix, range*sizeof(int));
  cudaMallocManaged(&tmp, range*sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");


  init<<<1,range>>>(bucket);
  cudaDeviceSynchronize();
  reduction<<<1,n>>>(bucket, key);
  cudaDeviceSynchronize();
  scan<<<1,range>>>(bucket, prefix, tmp, range);
  cudaDeviceSynchronize();
  writeback<<<1,range>>>(key, prefix);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
  cudaFree(prefix);
  cudaFree(tmp);
}
