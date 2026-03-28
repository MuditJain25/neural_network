#include "network.h"
#include <cmath>
#include <random>

void init_weights(SharedMemoryData* shm, bool is_master) {
    if (!is_master) return;

    std::mt19937 gen(42);
    std::normal_distribution<float> d1(0.0f, std::sqrt(2.0f / INPUT_SIZE)); // He intialization
    std::normal_distribution<float> d2(0.0f, std::sqrt(2.0f / HIDDEN_SIZE));

    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            shm->W1[i][j] = d1(gen);
        }
        shm->b1[i] = 0.0f;
    }

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            shm->W2[i][j] = d2(gen);
        }
        shm->b2[i] = 0.0f;
    }
}

void execute_forward_pass(SharedMemoryData* shm, const SharedDataset* ds, int sample_idx, int proc_id, int num_procs) {
    const uint8_t* img = ds->images + (sample_idx * MNIST_IMAGE_SIZE);

    // SLICING LOGIC: Determine bounds for this specific process
    int hidden_slice = HIDDEN_SIZE / num_procs;
    int h_start = proc_id * hidden_slice;
    int h_end = (proc_id == num_procs - 1) ? HIDDEN_SIZE : (h_start + hidden_slice);

    // --- LAYER 1: Forward Propagation ---
    for (int i = h_start; i < h_end; ++i) {
        float sum = shm->b1[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            sum += shm->W1[i][j] * (img[j] / 255.0f);
        }
        shm->Z1[i] = sum;
        shm->A1[i] = sum > 0 ? sum : 0; // ReLU Activation
    }

    // LAYER 1 BARRIER -> All processes must finish hidden layer before moving to output layer
    pthread_barrier_wait(&shm->barrier);

    // --- LAYER 2: Forward Propagation ---
    int out_slice = OUTPUT_SIZE / num_procs;
    int o_start = proc_id * out_slice;
    int o_end = (proc_id == num_procs - 1) ? OUTPUT_SIZE : (o_start + out_slice);

    for (int i = o_start; i < o_end; ++i) {
        float sum = shm->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            sum += shm->W2[i][j] * shm->A1[j]; // Uses full A1 safely!
        }
        shm->Z2[i] = sum;
    }

    // LAYER 2 BARRIER -> Wait for output dense to finish mapping
    pthread_barrier_wait(&shm->barrier);

    // SOFTMAX
    if (proc_id == 0) {
        float max_z = shm->Z2[0];
        for (int i = 1; i < OUTPUT_SIZE; ++i) {
            if (shm->Z2[i] > max_z) max_z = shm->Z2[i];
        }
        float sum_exp = 0;
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            shm->A2[i] = expl(shm->Z2[i] - max_z);
            sum_exp += shm->A2[i];
        }
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            shm->A2[i] /= sum_exp;
        }
    }

    // Wait for P0 to normalize the softmax probabilities
    pthread_barrier_wait(&shm->barrier);
}

void execute_backward_pass(SharedMemoryData* shm, const SharedDataset* ds, int sample_idx, int proc_id, int num_procs, float learning_rate) {
    const uint8_t* img = ds->images + (sample_idx * MNIST_IMAGE_SIZE);
    int label = ds->labels[sample_idx];

    // --- LOSS & OUTPUT GRADIENT ---
    if (proc_id == 0) {
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            float y = (i == label) ? 1.0f : 0.0f;
            shm->dZ2[i] = shm->A2[i] - y; // Derivative of Cross-Entropy + Softmax
        }
        
        // Compute Argmax for accuracy and log loss safely with MUTEX
        int best_pred = 0;
        float highest_prob = shm->A2[0];
        for (int i = 1; i < OUTPUT_SIZE; ++i) {
            if (shm->A2[i] > highest_prob) { highest_prob = shm->A2[i]; best_pred = i; }
        }

        pthread_mutex_lock(&shm->mutex);
        shm->epoch_loss += -logf(shm->A2[label] + 1e-8f); // Cross-entropy single sample
        if (best_pred == label) {
            shm->correct_predictions++;
        }
        pthread_mutex_unlock(&shm->mutex);
    }
    
    // BARRIER: Wait for Error computation to complete
    pthread_barrier_wait(&shm->barrier);

    // --- LAYER 2 BACKWARD ---
    int out_slice = OUTPUT_SIZE / num_procs;
    int o_start = proc_id * out_slice;
    int o_end = (proc_id == num_procs - 1) ? OUTPUT_SIZE : (o_start + out_slice);

    for (int i = o_start; i < o_end; ++i) {
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            shm->dW2[i][j] = shm->dZ2[i] * shm->A1[j];
        }
        shm->db2[i] = shm->dZ2[i];
    }

    // --- LAYER 1 BACKWARD ---
    int hidden_slice = HIDDEN_SIZE / num_procs;
    int h_start = proc_id * hidden_slice;
    int h_end = (proc_id == num_procs - 1) ? HIDDEN_SIZE : (h_start + hidden_slice);

    for (int j = h_start; j < h_end; ++j) {
        float err = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            err += shm->W2[i][j] * shm->dZ2[i]; // W2 read correctly here as it hasn't been updated yet!
        }
        shm->dZ1[j] = err * (shm->Z1[j] > 0 ? 1.0f : 0.0f); // ReLU Derivative
        
        for (int k = 0; k < INPUT_SIZE; ++k) {
            shm->dW1[j][k] = shm->dZ1[j] * (img[k] / 255.0f);
        }
        shm->db1[j] = shm->dZ1[j];
    }

    // BARRIER: Wait till ALL backward passes & gradients are recorded
    pthread_barrier_wait(&shm->barrier);

    // --- WEIGHT OPTIMIZATION (Stochastic Gradient Descent) ---
    // Every process applies updates for its slice.
    // They update Weights directly without lock, since slices are disjoint (No race condition!)
    
    for (int i = o_start; i < o_end; ++i) {
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            shm->W2[i][j] -= learning_rate * shm->dW2[i][j];
        }
        shm->b2[i] -= learning_rate * shm->db2[i];
    }
    
    for (int j = h_start; j < h_end; ++j) {
        for (int k = 0; k < INPUT_SIZE; ++k) {
            shm->W1[j][k] -= learning_rate * shm->dW1[j][k];
        }
        shm->b1[j] -= learning_rate * shm->db1[j];
    }

    // BARRIER: Finish SGD step
    pthread_barrier_wait(&shm->barrier);
}
