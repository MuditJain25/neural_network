#include "network.h"
#include <cmath>
#include <random>

void init_weights(SharedMemoryData* shm, bool is_master) {
    if (!is_master) return;

    std::mt19937 gen(42);

    for (int l = 0; l < NUM_LAYERS - 1; ++l) {
        int in_size = LAYER_SIZES[l];
        int out_size = LAYER_SIZES[l + 1];
        int w_offset = WEIGHT_OFFSETS[l];
        int b_offset = BIAS_OFFSETS[l];

        std::normal_distribution<float> d(0.0f, std::sqrt(2.0f / in_size)); // He initialization
        
        for (int i = 0; i < out_size; ++i) {
            for (int j = 0; j < in_size; ++j) {
                shm->W[w_offset + i * in_size + j] = d(gen);
            }
            shm->b[b_offset + i] = 0.0f;
        }
    }
}

void execute_forward_pass(SharedMemoryData* shm, const SharedDataset* ds, int sample_idx, int proc_id, int num_procs) {
    const uint8_t* img = ds->images + (sample_idx * MNIST_IMAGE_SIZE);

    // Initial layer A0 setup
    if (proc_id == 0) {
        int a_offset = NODE_OFFSETS[0];
        for (int j = 0; j < LAYER_SIZES[0]; ++j) {
            shm->A[a_offset + j] = img[j] / 255.0f;
        }
    }
    // Wait for A0 mapping
    pthread_barrier_wait(&shm->barrier);

    for (int l = 0; l < NUM_LAYERS - 1; ++l) {
        int in_size = LAYER_SIZES[l];
        int out_size = LAYER_SIZES[l + 1];
        
        int a_in_offset = NODE_OFFSETS[l];
        int a_out_offset = NODE_OFFSETS[l + 1];
        int w_offset = WEIGHT_OFFSETS[l];
        int b_offset = BIAS_OFFSETS[l];
        
        int slice = out_size / num_procs;
        int start = proc_id * slice;
        int end = (proc_id == num_procs - 1) ? out_size : (start + slice);

        for (int i = start; i < end; ++i) {
            float sum = shm->b[b_offset + i];
            for (int j = 0; j < in_size; ++j) {
                sum += shm->W[w_offset + i * in_size + j] * shm->A[a_in_offset + j];
            }
            shm->Z[a_out_offset + i] = sum;
            
            // Activation - ReLU for hidden layers, linear for output (Softmax handled later)
            if (l < NUM_LAYERS - 2) {
                shm->A[a_out_offset + i] = sum > 0 ? sum : 0; 
            }
        }
        pthread_barrier_wait(&shm->barrier);
    }

    // SOFTMAX on last layer
    if (proc_id == 0) {
        int l = NUM_LAYERS - 2;
        int out_size = LAYER_SIZES[l + 1];
        int a_out_offset = NODE_OFFSETS[l + 1];
        
        float max_z = shm->Z[a_out_offset + 0];
        for (int i = 1; i < out_size; ++i) {
            if (shm->Z[a_out_offset + i] > max_z) max_z = shm->Z[a_out_offset + i];
        }
        float sum_exp = 0;
        for (int i = 0; i < out_size; ++i) {
            shm->A[a_out_offset + i] = expl(shm->Z[a_out_offset + i] - max_z);
            sum_exp += shm->A[a_out_offset + i];
        }
        for (int i = 0; i < out_size; ++i) {
            shm->A[a_out_offset + i] /= sum_exp;
        }
    }
    pthread_barrier_wait(&shm->barrier);
}

void execute_backward_pass(SharedMemoryData* shm, const SharedDataset* ds, int sample_idx, int proc_id, int num_procs, float learning_rate) {
    int label = ds->labels[sample_idx];

    // --- LOSS & OUTPUT GRADIENT ---
    if (proc_id == 0) {
        int l = NUM_LAYERS - 2;
        int out_size = LAYER_SIZES[l + 1];
        int a_out_offset = NODE_OFFSETS[l + 1];
        
        for (int i = 0; i < out_size; ++i) {
            float y = (i == label) ? 1.0f : 0.0f;
            shm->dZ[a_out_offset + i] = shm->A[a_out_offset + i] - y;
        }
        
        int best_pred = 0;
        float highest_prob = shm->A[a_out_offset + 0];
        for (int i = 1; i < out_size; ++i) {
            if (shm->A[a_out_offset + i] > highest_prob) { 
                highest_prob = shm->A[a_out_offset + i]; 
                best_pred = i; 
            }
        }

        pthread_mutex_lock(&shm->mutex);
        shm->epoch_loss += -logf(shm->A[a_out_offset + label] + 1e-8f); 
        if (best_pred == label) {
            shm->correct_predictions++;
        }
        pthread_mutex_unlock(&shm->mutex);
    }
    pthread_barrier_wait(&shm->barrier);

    // --- BACKWARD LOOP ---
    for (int l = NUM_LAYERS - 2; l >= 0; --l) {
        int in_size = LAYER_SIZES[l];
        int out_size = LAYER_SIZES[l + 1];
        
        int a_in_offset = NODE_OFFSETS[l];
        int a_out_offset = NODE_OFFSETS[l + 1];
        int w_offset = WEIGHT_OFFSETS[l];
        int b_offset = BIAS_OFFSETS[l];

        int out_slice = out_size / num_procs;
        int o_start = proc_id * out_slice;
        int o_end = (proc_id == num_procs - 1) ? out_size : (o_start + out_slice);

        // Gradients for current layer's weights and biases
        for (int i = o_start; i < o_end; ++i) {
            for (int j = 0; j < in_size; ++j) {
                shm->dW[w_offset + i * in_size + j] = shm->dZ[a_out_offset + i] * shm->A[a_in_offset + j];
            }
            shm->db[b_offset + i] = shm->dZ[a_out_offset + i];
        }

        // Gradients for previous layer's activations (only if not input layer)
        if (l > 0) {
            int in_slice = in_size / num_procs;
            int i_start = proc_id * in_slice;
            int i_end = (proc_id == num_procs - 1) ? in_size : (i_start + in_slice);

            for (int j = i_start; j < i_end; ++j) {
                float err = 0.0f;
                // Accumulate error from next layer
                for (int i = 0; i < out_size; ++i) {
                    err += shm->W[w_offset + i * in_size + j] * shm->dZ[a_out_offset + i];
                }
                shm->dZ[a_in_offset + j] = err * (shm->Z[a_in_offset + j] > 0 ? 1.0f : 0.0f); // ReLU Deriv
            }
        }
        // Barrier before moving to next previous layer to ensure dZ is fully written
        pthread_barrier_wait(&shm->barrier);
    }

    // --- WEIGHT OPTIMIZATION (Stochastic Gradient Descent) ---
    for (int l = 0; l < NUM_LAYERS - 1; ++l) {
        int in_size = LAYER_SIZES[l];
        int out_size = LAYER_SIZES[l + 1];
        
        int w_offset = WEIGHT_OFFSETS[l];
        int b_offset = BIAS_OFFSETS[l];

        int out_slice = out_size / num_procs;
        int o_start = proc_id * out_slice;
        int o_end = (proc_id == num_procs - 1) ? out_size : (o_start + out_slice);
        
        for (int i = o_start; i < o_end; ++i) {
            for (int j = 0; j < in_size; ++j) {
                shm->W[w_offset + i * in_size + j] -= learning_rate * shm->dW[w_offset + i * in_size + j];
            }
            shm->b[b_offset + i] -= learning_rate * shm->db[b_offset + i];
        }
    }
    pthread_barrier_wait(&shm->barrier);
}
