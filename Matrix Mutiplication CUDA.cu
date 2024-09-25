#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define SIZE 1000

__global__ void gpu_matrix_mult(int *a, int *b, int *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

__global__ void gpu_matrix_mult_transpose(int *a, int *bT, int *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < m && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * bT[col * n + i];
        }
        c[row * m + col] = sum;
    }
}

void cpu_matrix_mult(int **h_a, int **h_b, int **h_c, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            int tmp = 0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i][h] * h_b[h][j];
            }
            h_c[i][j] = tmp;
        }
    }
}

void cpu_matrix_mult_transpose(int **h_a, int **h_bT, int **h_c, int m, int n) {
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            int tmp = 0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i][h] * h_bT[j][h];
            }
            h_c[i][j] = tmp;
        }
    }
}

void transpose_matrix(int **B, int **B_T, int n, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            B_T[j][i] = B[i][j];  // Transpor a matriz B
        }
    }
}

int main(int argc, char const *argv[])
{
    int m = SIZE, n = SIZE, k = SIZE;
    srand(3333);

    int **h_a = (int **)malloc(SIZE * sizeof(int *));
    int **h_b = (int **)malloc(SIZE * sizeof(int *));
    int **h_bT = (int **)malloc(SIZE * sizeof(int *));
    int **h_c = (int **)malloc(SIZE * sizeof(int *));
    int **h_cc = (int **)malloc(SIZE * sizeof(int *));
    for (int i = 0; i < SIZE; i++) {
        h_a[i] = (int *)malloc(SIZE * sizeof(int));
        h_b[i] = (int *)malloc(SIZE * sizeof(int));
        h_bT[i] = (int *)malloc(SIZE * sizeof(int));
        h_c[i] = (int *)malloc(SIZE * sizeof(int));
        h_cc[i] = (int *)malloc(SIZE * sizeof(int));
    }

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            h_a[i][j] = rand() % 1024;
            h_b[i][j] = rand() % 1024;
        }
    }

    transpose_matrix(h_b, h_bT, n, k);

    clock_t start_cpu_normal = clock();
    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);
    clock_t end_cpu_normal = clock();
    double cpu_time_normal = ((double)(end_cpu_normal - start_cpu_normal)) / CLOCKS_PER_SEC;

    clock_t start_cpu_transpose = clock();
    cpu_matrix_mult_transpose(h_a, h_bT, h_c, m, n);
    clock_t end_cpu_transpose = clock();
    double cpu_time_transpose = ((double)(end_cpu_transpose - start_cpu_transpose)) / CLOCKS_PER_SEC;

    int *d_a, *d_b, *d_bT, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int) * m * n);
    cudaMalloc((void **) &d_b, sizeof(int) * n * k);
    cudaMalloc((void **) &d_bT, sizeof(int) * n * m);  // B^T agora é n * m
    cudaMalloc((void **) &d_c, sizeof(int) * m * k);

    int *h_a_linear = (int *)malloc(m * n * sizeof(int));
    int *h_b_linear = (int *)malloc(n * k * sizeof(int));
    int *h_bT_linear = (int *)malloc(n * m * sizeof(int));
    int *h_c_linear = (int *)malloc(m * k * sizeof(int));

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            h_a_linear[i * n + j] = h_a[i][j];

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j)
            h_b_linear[i * k + j] = h_b[i][j];

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            h_bT_linear[j * n + i] = h_bT[j][i];

    cudaMemcpy(d_a, h_a_linear, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b_linear, sizeof(int) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bT, h_bT_linear, sizeof(int) * n * m, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    cudaEventRecord(start);
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_normal = 0;
    cudaEventElapsedTime(&milliseconds_normal, start, stop);

    cudaEventRecord(start);
    gpu_matrix_mult_transpose<<<dimGrid, dimBlock>>>(d_a, d_bT, d_c, m, n, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds_transpose = 0;
    cudaEventElapsedTime(&milliseconds_transpose, start, stop);

    printf("Multiplicar as linhas da matriz A com a matriz B e salvar em C (CPU)\n");
    printf("Tempo de execucao: %f seconds\n", cpu_time_normal);
    printf("Multiplicar as linhas da matriz A com as linhas da matriz transposta B e salvar em C (CPU)\n");
    printf("Tempo de execucao: %f seconds\n", cpu_time_transpose);
    printf("Multiplicar as linhas da matriz A com a matriz B e salvar em C (GPU)\n");
    printf("Tempo de execucao: %f milliseconds\n", milliseconds_normal);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_bT);
    cudaFree(d_c);

    for (int i = 0; i < SIZE; i++) {
        free(h_a[i]);
        free(h_b[i]);
        free(h_bT[i]);
        free(h_c[i]);
        free(h_cc[i]);
    }
    free(h_a);
    free(h_b);
    free(h_bT);
    free(h_c);
    free(h_cc);

    free(h_a_linear);
    free(h_b_linear);
    free(h_bT_linear);
    free(h_c_linear);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}