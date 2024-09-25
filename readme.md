# Final Project for the Parallel Processing Course

**Federal University of Espírito Santo**

**Professor:** Guilherme Fernandes de Souza Miguel  
**Student:** Lázaro José Pedrosa dos Reis  

---

## Project Objective:

Develop a program that performs matrix multiplication in the following formats:

- Single-core matrix multiplication (with and without transposition) (CPU)
- Multithreading
- MPD (OpenMP)
- CUDA

### Single-core matrix multiplication (with and without transposition) (CPU) and CUDA
In this format, the program multiplies two matrices using a single CPU core, considering two approaches: direct multiplication and multiplication with matrix transposition for optimization. Additionally, the implementation includes matrix multiplication using CUDA (Compute Unified Device Architecture), where the operation is executed on a GPU, leveraging its parallel architecture to improve efficiency and execution time.

### Multithreading
Multithreaded matrix multiplication uses multiple threads to divide the workload across available CPU cores, enhancing efficiency and reducing execution time compared to the single-core approach.

### MPD (OpenMP)
The use of OpenMP allows the program to perform matrix multiplication in parallel within a C/C++ programming environment. Through compilation directives, OpenMP simplifies the implementation of parallelism in loops, making the most of the available hardware.

# Matrix Multiplication Results

## Results Obtained:

1. **Multiply matrix A rows with matrix B and store in C (CPU)**  
   Execution time: **4.453461 seconds**

2. **Multiply matrix A rows with transposed matrix B and store in C (CPU)**  
   Execution time: **3.019492 seconds**

3. **Multiply matrix A rows with matrix B and store in C (GPU)**  
   Execution time: **7.277696 milliseconds**

4. **Multiply matrix A rows with matrix B and store in C (MULTITHREAD)**  
   Execution time: **17.5007 seconds**

5. **Multiply matrix A rows with matrix B and store in C (OMP)**  
   Execution time: **5.80317 seconds**

---

## Results Comparison:

The comparison of execution times reveals significant differences between the approaches. Matrix multiplication using the GPU achieved the best performance, with an execution time of just **7.277696 milliseconds**, showcasing the efficiency of hardware-accelerated parallelism.

On the other hand, the multithreaded approach was the slowest, with a time of **17.5007 seconds**, likely due to the overhead of thread management compared to single-core execution.

Matrix multiplication with transposition showed improvement over direct multiplication, with a time of **3.019492 seconds**, indicating that memory access optimization can positively impact performance. OpenMP also demonstrated an improvement over single-core multiplication, with a time of **5.80317 seconds**.

These results suggest that when dealing with matrix multiplication, utilizing a GPU or optimization techniques like transposition and OpenMP can provide significant performance gains.
