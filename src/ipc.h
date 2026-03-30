#pragma once

#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <array>

// --- Flexible Compile-Time Architecture Config ---
// ADD/REMOVE/RESIZE LAYERS HERE:
constexpr std::array<int, 4> LAYER_SIZES = {784, 256, 128, 10};
constexpr int NUM_LAYERS = LAYER_SIZES.size();

// Compile-Time Helper: Calculate total weights across all layers
constexpr int get_total_weights() {
    int total = 0;
    for (int i = 0; i < NUM_LAYERS - 1; ++i) {
        total += LAYER_SIZES[i] * LAYER_SIZES[i + 1];
    }
    return total;
}

// Compile-Time Helper: Calculate total biases across all layers
constexpr int get_total_biases() {
    int total = 0;
    for (int i = 0; i < NUM_LAYERS - 1; ++i) {
        total += LAYER_SIZES[i + 1];
    }
    return total;
}

// Compile-Time Helper: Calculate total node activations across all layers
constexpr int get_total_nodes() {
    int total = 0;
    for (int i = 0; i < NUM_LAYERS; ++i) {
        total += LAYER_SIZES[i];
    }
    return total;
}

// Constexpr Offset Tables for fast runtime indexing
constexpr std::array<int, NUM_LAYERS> get_weight_offsets() {
    std::array<int, NUM_LAYERS> offsets = {0};
    int current = 0;
    for (int i = 0; i < NUM_LAYERS - 1; ++i) {
        offsets[i] = current;
        current += LAYER_SIZES[i] * LAYER_SIZES[i + 1];
    }
    return offsets;
}

constexpr std::array<int, NUM_LAYERS> get_bias_offsets() {
    std::array<int, NUM_LAYERS> offsets = {0};
    int current = 0;
    for (int i = 0; i < NUM_LAYERS - 1; ++i) {
        offsets[i] = current;
        current += LAYER_SIZES[i + 1];
    }
    return offsets;
}

constexpr std::array<int, NUM_LAYERS> get_node_offsets() {
    std::array<int, NUM_LAYERS> offsets = {0};
    int current = 0;
    for (int i = 0; i < NUM_LAYERS; ++i) {
        offsets[i] = current;
        current += LAYER_SIZES[i];
    }
    return offsets;
}

constexpr int TOTAL_WEIGHTS = get_total_weights();
constexpr int TOTAL_BIASES = get_total_biases();
constexpr int TOTAL_NODES = get_total_nodes();
constexpr auto WEIGHT_OFFSETS = get_weight_offsets();
constexpr auto BIAS_OFFSETS = get_bias_offsets();
constexpr auto NODE_OFFSETS = get_node_offsets();

// Memory layout for Shared Memory Segment
struct SharedMemoryData {
    pthread_mutex_t mutex;
    pthread_barrier_t barrier;

    // Flat Weight and Bias Arrays
    float W[TOTAL_WEIGHTS];
    float b[TOTAL_BIASES];

    // Flat Gradient Accumulators
    float dW[TOTAL_WEIGHTS];
    float db[TOTAL_BIASES];

    // Network State
    // A contains activations (A[NODE_OFFSETS[0]] is Input, A[NODE_OFFSETS[1]] is layer 1...)
    float Z[TOTAL_NODES];
    float A[TOTAL_NODES];
    float dZ[TOTAL_NODES];

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
