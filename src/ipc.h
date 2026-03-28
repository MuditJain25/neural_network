#pragma once

#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdexcept>
#include <string>

// Neural Network Architecture Hyperparameters
#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10

// Memory layout for Shared Memory Segment
struct SharedMemoryData {
    pthread_mutex_t mutex;
    pthread_barrier_t barrier;

    // Weights and Biases
    float W1[HIDDEN_SIZE][INPUT_SIZE];
    float b1[HIDDEN_SIZE];
    
    float W2[OUTPUT_SIZE][HIDDEN_SIZE];
    float b2[OUTPUT_SIZE];

    // Gradient Accumulators
    float dW1[HIDDEN_SIZE][INPUT_SIZE];
    float db1[HIDDEN_SIZE];
    float dW2[OUTPUT_SIZE][HIDDEN_SIZE];
    float db2[OUTPUT_SIZE];

    // Forward Pass Variables
    float A0[INPUT_SIZE];    // Input layer
    float Z1[HIDDEN_SIZE];   // Hidden layer linear sum
    float A1[HIDDEN_SIZE];   // Hidden layer activation (ReLU)
    float Z2[OUTPUT_SIZE];   // Output layer linear sum
    float A2[OUTPUT_SIZE];   // Output layer activation (Softmax)

    // Backward Pass Errors
    float dZ2[OUTPUT_SIZE];
    float dZ1[HIDDEN_SIZE];

    // Training state and metrics
    int current_label;
    float epoch_loss;
    int correct_predictions;
    
    // Control Flags
    bool training_complete;
    int active_workers;
};

// IPC Utility functions
SharedMemoryData* init_shared_memory(const std::string& name, int num_workers, bool is_master);
void cleanup_shared_memory(const std::string& name, SharedMemoryData* shm_ptr);
