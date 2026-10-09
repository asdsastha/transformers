// Step 1: your first GPU program.
//
// Compile:  nvcc 01_hello.cu -o hello
// Run:      ./hello

#include <cstdio>

// A "kernel": a function that runs on the GPU.
// __global__ means "the CPU calls it, the GPU runs it".
// The GPU runs many copies of it at the same time, one per thread.
__global__ void hello()
{
    // Each copy can ask "which thread am I?"
    //   threadIdx.x = my number inside my block
    //   blockIdx.x  = which block I'm in
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    // Launch the kernel: <<<number of blocks, threads per block>>>
    // Here: 2 blocks x 4 threads = 8 copies of hello() running in parallel.
    hello<<<2, 4>>>();

    // The launch returns immediately; the CPU doesn't wait for the GPU.
    // This line makes the CPU wait until the GPU is finished,
    // otherwise the program could exit before anything is printed.
    cudaDeviceSynchronize();

    return 0;
}
