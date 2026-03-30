# Multiprocessing N-Layer Neural Network: Codebase Walkthrough

This codebase is a specialized implementation of a deep neural network training on the MNIST dataset using C++17. What makes this implementation remarkably unique is that it avoids traditional threading (like `std::thread` or OpenMP) in favor of **OS-level multiprocessing using `fork()`**, **POSIX Shared Memory**, and **process-shared synchronization primitives**. 

We recently upgraded the network from a hardcoded 2-layer model to a fully dynamic **N-Layer** architecture determined entirely at compile time using purely flat 1D memory mapping.

Here is a comprehensive breakdown of how the entire codebase orchestrates itself.

---

## 1. Multiprocessing Initialization (`src/main.cpp`)

When `./mlp_train` is run, the main operating system process (the "Master") prepares the memory environments.
1. **Load Dataset**: It parses the MNIST binary files and loads all 60,000 images seamlessly into a POSIX shared memory block (`/mnist_ds_shm`).
2. **Setup Network Memory**: It creates another shared memory block (`/mnist_net_shm`) dedicated entirely to our architectural math—weights, biases, and gradients.
3. **Forking**: The Master uses `fork()` to spawn `N` exact replicas of itself (Workers).
   - Because we use `fork()` combined with POSIX `mmap()` (MAP_SHARED), the entire virtual address space mapped by `mmap` is inherited perfectly. Pointers referencing inside the shared memory segment point to exactly the same identical physical pages across all processes. 

## 2. Compile-Time Architecture Config (`src/ipc.h`)

This is the brain behind the new flexible architecture.

```cpp
constexpr std::array<int, 4> LAYER_SIZES = {784, 256, 128, 10};
constexpr int NUM_LAYERS = LAYER_SIZES.size();
```
We use C++17 `constexpr` capabilities to define our entire neural network structure as an unchangeable array. When you compile the code, the compiler literally runs the helper functions in `ipc.h` to calculate the **exact number of floats** needed for every single weight matrix, bias, and gradient accumulator.

`SharedMemoryData` uses these compile-time variables to create tightly packed, 1D inline arrays:
```cpp
float W[TOTAL_WEIGHTS];
float Z[TOTAL_NODES]; // etc...
```
This guarantees an insanely fast contiguous memory block with no dynamic `malloc` overhead that perfectly fits the mapped shared memory page!

## 3. Parallel Execution Slicing (`src/network.cpp`)

During training, all forked OS processes jump into `execute_forward_pass` and `execute_backward_pass` simultaneously.

To ensure processes don't calculate the same exact math (or overwrite each other), the work is heavily **sliced**.
```cpp
int slice = out_size / num_procs;
int start = proc_id * slice;
int end = (proc_id == num_procs - 1) ? out_size : (start + slice);
```
In a hidden layer with 256 nodes split across 4 processes:
- `Process 0` computes nodes 0 -> 63
- `Process 1` computes nodes 64 -> 127
- ... and so on.

Because they map outputs into **disjoint** exact slots in `shm->W` and `shm->Z`, there is actually absolutely **no race condition** during vector calculations, which means we avoid the heavy overhead of `pthread_mutex_lock` during the vast majority of the workload.

> [!TIP]
> This is why using `flat 1D arrays` in `SharedMemoryData` makes mapping calculations faster. Memory hops are sequentially predictable for CPU caches!

## 4. Cross-Process Synchronization (`src/ipc.cpp`)

To maintain mathematical accuracy, Process 0 cannot be doing layer 3 calculations while Process 2 is still working on layer 1. The codebase uses **POSIX Barriers**:
```cpp
pthread_barrierattr_setpshared(&b_attr, PTHREAD_PROCESS_SHARED);
pthread_barrier_init(&shm->barrier, &b_attr, num_workers);
```
Using `PTHREAD_PROCESS_SHARED` allows the barrier to work across different OS processes (not just threads).

In `network.cpp`, after every single layer is computed inside the `for (int l = 0; l < NUM_LAYERS - 1; ++l)` loop, we call `pthread_barrier_wait(&shm->barrier);`. This forces fast worker processes to completely halt until all other processes reach the exact same line, guaranteeing Layer 1 is mathematically perfect before anyone starts Layer 2.

## 5. Metrics Tracking 

Since worker processes are strictly computing gradients inside their exact sliced partitions, who tracks Accuracy and Loss?

In both the forward and backward passes, **Process `id == 0` is designated as the Orchestrator for global functions**:
1. Applying the mathematical `Softmax` on the final output layer prediction outputs.
2. Tracking `epoch_loss` and validating predictions against ground-truth datasets.

Because metrics are global accumulators, `Process 0` locks a **Process-Shared Mutex** (`pthread_mutex_lock(&shm->mutex)`) to ensure nothing alters the variables while they log the data.

## 6. Teardown

Once all epochs finish, the `fork()` children natively `exit(0)`.
The Master process relies on standard OS `waitpid()` blocking to catch all children as they cleanly shutdown, resolving their states. Finally, it surgically disassembles the memory structures by destroying the mutexes and `munmap()`ing the POSIX blocks.
