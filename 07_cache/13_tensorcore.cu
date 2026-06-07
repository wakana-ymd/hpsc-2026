#include <chrono>
#include <cmath>
#include <cublas_v2.h>
#include <iostream>
#include <mma.h>
#include <random>
#include <stdint.h>

using namespace std;
using namespace nvcuda;

__global__ void kernel_bpack_128x128_8w(int dim_m, int dim_k,
                                        const half *__restrict__ d_a,
                                        const half *__restrict__ d_b_packed,
                                        float *__restrict__ d_c) {
  constexpr int CTA_M = 128;
  constexpr int CTA_N = 128;
  constexpr int K_TILE = 16;
  constexpr int A_LOADER_THREADS = 128;
  constexpr int B_LOADER_THREADS = 128;
  constexpr int A_VECS_PER_ROW = CTA_M / 8;
  constexpr int A_SEGMENTS = K_TILE * A_VECS_PER_ROW;
  constexpr int A_LOADS = A_SEGMENTS / A_LOADER_THREADS;
  constexpr int B_SEGMENTS = CTA_N * 2;
  constexpr int B_LOADS = B_SEGMENTS / B_LOADER_THREADS;

  int offset_a_m = CTA_M * blockIdx.x;
  int tile_n = blockIdx.y;
  int offset_b_n = CTA_N * tile_n;
  int i = threadIdx.x;
  int warp_id = i / 32;
  int warp_row = warp_id / 2;
  int warp_col = warp_id % 2;
  int k_tiles = dim_k / K_TILE;

  __shared__ alignas(16) half block_a[2][K_TILE][CTA_M];
  __shared__ alignas(16) half block_b[2][CTA_N][K_TILE];

  if (i < A_LOADER_THREADS) {
    #pragma unroll
    for (int l = 0; l < A_LOADS; l++) {
      int seg = i + l * A_LOADER_THREADS;
      int row = seg / A_VECS_PER_ROW;
      int vec = seg % A_VECS_PER_ROW;
      reinterpret_cast<int4 *>(&block_a[0][row][0])[vec] =
          reinterpret_cast<const int4 *>(d_a + row * dim_m + offset_a_m)[vec];
      if (k_tiles > 1) {
        reinterpret_cast<int4 *>(&block_a[1][row][0])[vec] =
            reinterpret_cast<const int4 *>(d_a + (K_TILE + row) * dim_m + offset_a_m)[vec];
      }
    }
  } else {
    int loader = i - A_LOADER_THREADS;
    #pragma unroll
    for (int l = 0; l < B_LOADS; l++) {
      int seg = loader + l * B_LOADER_THREADS;
      int col = seg / 2;
      int vec = seg % 2;
      const half *tile_ptr = d_b_packed + (((tile_n * k_tiles) + 0) * CTA_N + col) * K_TILE;
      reinterpret_cast<int4 *>(&block_b[0][col][0])[vec] =
          reinterpret_cast<const int4 *>(tile_ptr)[vec];
      if (k_tiles > 1) {
        tile_ptr = d_b_packed + (((tile_n * k_tiles) + 1) * CTA_N + col) * K_TILE;
        reinterpret_cast<int4 *>(&block_b[1][col][0])[vec] =
            reinterpret_cast<const int4 *>(tile_ptr)[vec];
      }
    }
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 4; c++) {
      wmma::fill_fragment(acc[r][c], 0.0f);
    }
  }

  for (int tile = 0; tile < k_tiles; tile++) {
    int stage = tile & 1;
    int next_stage = stage ^ 1;
    __syncthreads();

    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[4];
    #pragma unroll 4
    for (int c = 0; c < 4; c++) {
      wmma::load_matrix_sync(b_frag[c], &block_b[stage][warp_col * 64 + c * 16][0], 16);
    }

    int4 next_a[A_LOADS];
    int4 next_b[B_LOADS];
    int next_tile = tile + 1;
    int next_k = next_tile * K_TILE;
    bool has_next = next_tile < k_tiles;

    #pragma unroll 2
    for (int r = 0; r < 2; r++) {
      int row_tile = warp_row * 2 + r;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::load_matrix_sync(a_frag, &block_a[stage][0][row_tile * 16], CTA_M);
      if (r == 0 && has_next) {
        if (i < A_LOADER_THREADS) {
          #pragma unroll
          for (int l = 0; l < A_LOADS; l++) {
            int seg = i + l * A_LOADER_THREADS;
            int row = seg / A_VECS_PER_ROW;
            int vec = seg % A_VECS_PER_ROW;
            next_a[l] =
                reinterpret_cast<const int4 *>(d_a + (next_k + row) * dim_m + offset_a_m)[vec];
          }
        } else {
          int loader = i - A_LOADER_THREADS;
          #pragma unroll
          for (int l = 0; l < B_LOADS; l++) {
            int seg = loader + l * B_LOADER_THREADS;
            int col = seg / 2;
            int vec = seg % 2;
            const half *tile_ptr =
                d_b_packed + (((tile_n * k_tiles) + next_tile) * CTA_N + col) * K_TILE;
            next_b[l] = reinterpret_cast<const int4 *>(tile_ptr)[vec];
          }
        }
      }
      #pragma unroll 4
      for (int c = 0; c < 4; c++) {
        wmma::mma_sync(acc[r][c], a_frag, b_frag[c], acc[r][c]);
      }
    }

    if (has_next) {
      if (i < A_LOADER_THREADS) {
        #pragma unroll
        for (int l = 0; l < A_LOADS; l++) {
          int seg = i + l * A_LOADER_THREADS;
          int row = seg / A_VECS_PER_ROW;
          int vec = seg % A_VECS_PER_ROW;
          reinterpret_cast<int4 *>(&block_a[next_stage][row][0])[vec] = next_a[l];
        }
      } else {
        int loader = i - A_LOADER_THREADS;
        #pragma unroll
        for (int l = 0; l < B_LOADS; l++) {
          int seg = loader + l * B_LOADER_THREADS;
          int col = seg / 2;
          int vec = seg % 2;
          reinterpret_cast<int4 *>(&block_b[next_stage][col][0])[vec] = next_b[l];
        }
      }
    }
  }

  #pragma unroll 2
  for (int r = 0; r < 2; r++) {
    #pragma unroll 4
    for (int c = 0; c < 4; c++) {
      int c_m = offset_a_m + (warp_row * 2 + r) * 16;
      int c_n = offset_b_n + warp_col * 64 + c * 16;
      wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
    }
  }
}

void check_cuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    cerr << msg << ": " << cudaGetErrorString(err) << "\n";
    exit(1);
  }
}

double compute_error(const float *ref, const float *test, int m, int n) {
  double err = 0.0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      err += fabs(ref[m * i + j] - test[m * i + j]);
    }
  }
  return err / n / m;
}

int main(int argc, const char **argv) {
  constexpr int CTA_M = 128;
  constexpr int CTA_N = 128;
  constexpr int K_TILE = 16;

  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0f;
  float beta = 0.0f;
  int Nt = 10;
  if ((m % CTA_M) != 0 || (n % CTA_N) != 0 || (k % K_TILE) != 0) {
    cerr << "This kernel assumes m is a multiple of 128, n is a multiple of 128, and k is a multiple of 16.\n";
    return 1;
  }

  int n_tiles = n / CTA_N;
  int k_tiles = k / K_TILE;
  size_t size_a = size_t(m) * k;
  size_t size_b = size_t(k) * n;
  size_t size_c = size_t(m) * n;
  size_t bytes_a = size_a * sizeof(float);
  size_t bytes_b = size_b * sizeof(float);
  size_t bytes_c = size_c * sizeof(float);
  size_t bytes_a_half = size_a * sizeof(half);
  size_t bytes_b_half = size_b * sizeof(half);
  size_t bytes_b_packed = size_b * sizeof(half);

  float *A, *B, *C, *C2;
  half *A_half, *B_half, *B_half_packed;
  float *h_A = new float[size_a];
  float *h_B = new float[size_b];
  half *h_A_half = new half[size_a];
  half *h_B_half = new half[size_b];
  half *h_B_half_packed = new half[size_b];

  cudaMalloc(&A, bytes_a);
  cudaMalloc(&B, bytes_b);
  cudaMalloc(&C, bytes_c);
  cudaMalloc(&C2, bytes_c);
  cudaMalloc(&A_half, bytes_a_half);
  cudaMalloc(&B_half, bytes_b_half);
  cudaMalloc(&B_half_packed, bytes_b_packed);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      h_A[k * i + j] = drand48();
      h_A_half[i * k + j] = __float2half(h_A[k * i + j]);
    }
  }
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      h_B[n * i + j] = drand48();
      h_B_half[i * n + j] = __float2half(h_B[n * i + j]);
    }
  }

  for (int nt = 0; nt < n_tiles; nt++) {
    for (int kt = 0; kt < k_tiles; kt++) {
      for (int col = 0; col < CTA_N; col++) {
        for (int row = 0; row < K_TILE; row++) {
          size_t dst = (((size_t(nt) * k_tiles + kt) * CTA_N + col) * K_TILE + row);
          size_t src = size_t(nt * CTA_N + col) * k + (kt * K_TILE + row);
          h_B_half_packed[dst] = h_B_half[src];
        }
      }
    }
  }

  check_cuda(cudaMemcpy(A, h_A, bytes_a, cudaMemcpyHostToDevice), "cudaMemcpy A failed");
  check_cuda(cudaMemcpy(B, h_B, bytes_b, cudaMemcpyHostToDevice), "cudaMemcpy B failed");
  check_cuda(cudaMemcpy(A_half, h_A_half, bytes_a_half, cudaMemcpyHostToDevice), "cudaMemcpy A_half failed");
  check_cuda(cudaMemcpy(B_half, h_B_half, bytes_b_half, cudaMemcpyHostToDevice), "cudaMemcpy B_half failed");
  check_cuda(cudaMemcpy(B_half_packed, h_B_half_packed, bytes_b_packed, cudaMemcpyHostToDevice),
             "cudaMemcpy B_half_packed failed");
  check_cuda(cudaMemset(C, 0, bytes_c), "cudaMemset C failed");
  check_cuda(cudaMemset(C2, 0, bytes_c), "cudaMemset C2 failed");

  delete[] h_A;
  delete[] h_B;
  delete[] h_A_half;
  delete[] h_B_half;
  delete[] h_B_half_packed;

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt + 2; i++) {
    if (i == 2) {
      tic = chrono::steady_clock::now();
    }
    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_N,
                 CUBLAS_OP_N,
                 m,
                 n,
                 k,
                 &alpha,
                 A, CUDA_R_32F, m,
                 B, CUDA_R_32F, k,
                 &beta,
                 C, CUDA_R_32F, m,
                 CUBLAS_COMPUTE_32F_FAST_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    check_cuda(cudaDeviceSynchronize(), "cublas synchronize failed");
  }
  auto toc = chrono::steady_clock::now();

  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;

  dim3 block = dim3(256);
  dim3 grid = dim3(m / CTA_M, n / CTA_N);
  for (int i = 0; i < Nt + 2; i++) {
    if (i == 2) {
      tic = chrono::steady_clock::now();
    }
    kernel_bpack_128x128_8w<<<grid, block>>>(m, k, A_half, B_half_packed, C2);
    check_cuda(cudaGetLastError(), "kernel launch failed");
    check_cuda(cudaDeviceSynchronize(), "kernel execution failed");
  }
  toc = chrono::steady_clock::now();
  double tcustom = chrono::duration<double>(toc - tic).count() / Nt;
  double custom_flops = double(num_flops) / tcustom / 1.0e9;

  float *h_C = new float[size_c];
  float *h_C2 = new float[size_c];
  check_cuda(cudaMemcpy(h_C, C, bytes_c, cudaMemcpyDeviceToHost), "cudaMemcpy C failed");
  check_cuda(cudaMemcpy(h_C2, C2, bytes_c, cudaMemcpyDeviceToHost), "cudaMemcpy C2 failed");
  double err = compute_error(h_C, h_C2, m, n);

  printf("CUBLAS: %.2f Gflops, CUSTOM_BPACK: %.2f Gflops\n", cublas_flops, custom_flops);
  printf("error: %lf\n", err);
  printf("note: B pack cost is one-time and not included in CUSTOM_BPACK timing.\n");

  delete[] h_C;
  delete[] h_C2;
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cudaFree(A_half);
  cudaFree(B_half);
  cudaFree(B_half_packed);
  cublasDestroy(cublas_handle);
}
