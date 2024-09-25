#include <iostream>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <vector>

#define SIZE 1000

void multiply_matrices(int** A, int** B, int** C, int start_row, int end_row) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

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

    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    clock_t start_time = clock();

    int rows_per_thread = SIZE / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? SIZE : start_row + rows_per_thread;
        threads.push_back(std::thread(multiply_matrices, A, B_transpose, C, start_row, end_row));
    }

    for (auto& th : threads) {
        th.join();
    }

    clock_t end_time = clock();
    double cpu_time_normal = double(end_time - start_time) / CLOCKS_PER_SEC;

    std::cout << "Multiplicar as linhas da matriz A com a matriz B e salvar em C (MULTITHREAD)\n";
    std::cout << "Tempo de execução: " << cpu_time_normal << " seconds\n";

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
