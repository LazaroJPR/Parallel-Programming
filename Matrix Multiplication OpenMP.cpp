#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <chrono>

#define SIZE 1000

int main() {
    int **A = (int **)malloc(SIZE * sizeof(int *));
    int **B = (int **)malloc(SIZE * sizeof(int *));
    int **B_transpose = (int **)malloc(SIZE * sizeof(int *));
    int **C = (int **)malloc(SIZE * sizeof(int *));
    for (int i = 0; i < SIZE; i++) {
        A[i] = (int *)malloc(SIZE * sizeof(int));
        B[i] = (int *)malloc(SIZE * sizeof(int));
        B_transpose[i] = (int *)malloc(SIZE * sizeof(int));
        C[i] = (int *)malloc(SIZE * sizeof(int));
    }

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = i * SIZE + j;
            B[i][j] = i * SIZE + j;
        }
    }

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            B_transpose[j][i] = B[i][j];
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B_transpose[j][k];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time_normal = end - start;

    std::cout << "Multiplicar as linhas da matriz A com a matriz B e salvar em C (OMP)\n";
    std::cout << "Tempo de execução: " << cpu_time_normal.count() << " seconds\n";

    for (int i = 0; i < SIZE; i++) {
        free(A[i]);
        free(B[i]);
        free(B_transpose[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(B_transpose);
    free(C);

    return 0;
}
