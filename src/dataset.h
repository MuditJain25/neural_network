#pragma once
#include "ipc.h"
#include <string>

#define MNIST_IMAGE_SIZE 784

// Complete shared dataset memory architecture
struct SharedDataset {
    int num_samples;
    uint8_t* images; // Stored linearly: num_samples * 784
    uint8_t* labels; // Stored linearly: num_samples
};

SharedDataset init_shared_dataset(const std::string& shm_name, int max_samples, bool is_master);
void load_mnist_multithreaded(const std::string& image_path, const std::string& label_path, SharedDataset& dataset, int num_threads);
void cleanup_shared_dataset(const std::string& shm_name, SharedDataset& dataset);
