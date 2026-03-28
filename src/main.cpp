#include "ipc.h"
#include "dataset.h"
#include "network.h"
#include <iostream>
#include <sys/wait.h>
#include <vector>
#include <chrono>

int main() {
    int num_procs = 4;
    int epochs = 5;
    float learning_rate = 0.01f;
    
    // POSIX shared memory object names
    std::string shm_net_name = "/mnist_net_shm";
    std::string shm_ds_name = "/mnist_ds_shm";

    std::cout << "Loading dataset into shared memory..." << std::endl;
    SharedDataset ds;
    try {
        ds = init_shared_dataset(shm_ds_name, 60000, true);
        load_mnist_multithreaded("train-images-idx3-ubyte", "train-labels-idx1-ubyte", ds, 4);
        std::cout << "Loaded " << ds.num_samples << " samples." << std::endl;
    } catch(const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        std::cerr << "Please ensure 'train-images-idx3-ubyte' and 'train-labels-idx1-ubyte' exist in the directory." << std::endl;
        cleanup_shared_dataset(shm_ds_name, ds);
        return 1;
    }

    std::cout << "Initializing shared memory for network..." << std::endl;
    SharedMemoryData* shm;
    try {
        shm = init_shared_memory(shm_net_name, num_procs, true);
    } catch(const std::exception& e) {
        std::cerr << "Error initializing network shared memory: " << e.what() << std::endl;
        return 1;
    }
    
    init_weights(shm, true);

    std::vector<pid_t> pids;
    std::cout << "Forking " << num_procs << " process slices..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int p = 0; p < num_procs; ++p) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process: Each calculates for its designated neuron slice
            // Re-open shared memories in case memory map is not perfectly preserved (though fork should preserve it)
            // But we don't strictly need to reopen them, fork gives same pointer mapping! COWs apply to private, but MAP_SHARED stays shared!
            
            for (int epoch = 0; epoch < epochs; ++epoch) {
                for (int sample = 0; sample < ds.num_samples; ++sample) {
                    execute_forward_pass(shm, &ds, sample, p, num_procs);
                    execute_backward_pass(shm, &ds, sample, p, num_procs, learning_rate);
                }

                // Process 0 tracks the overall epoch loss and resets
                if (p == 0) {
                    std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                              << " - Loss: " << (shm->epoch_loss / ds.num_samples)
                              << " - Accuracy: " << (shm->correct_predictions * 100.0f / ds.num_samples) << "%" << std::endl;
                    shm->epoch_loss = 0.0f;
                    shm->correct_predictions = 0;
                }
                
                // Epoch Barrier: prevent workers from bleeding into next epoch before P0 resets metrics
                pthread_barrier_wait(&shm->barrier);
            }
            return 0; // Worker safely exits
        } else if (pid > 0) {
            pids.push_back(pid); // Master records worker
        } else {
            std::cerr << "Fork failed for process " << p << std::endl;
            return 1;
        }
    }

    // Master Process acts as the orchestrator manager
    for (pid_t pid : pids) {
        waitpid(pid, nullptr, 0);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Training complete in " << diff.count() << " seconds." << std::endl;

    // Zero-Copy teardown
    cleanup_shared_memory(shm_net_name, shm);
    cleanup_shared_dataset(shm_ds_name, ds);

    return 0;
}
