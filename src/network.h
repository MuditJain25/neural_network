#pragma once
#include "ipc.h"
#include "dataset.h"

// Neural Network Distributed Core functions
void init_weights(SharedMemoryData* shm, bool is_master);
void execute_forward_pass(SharedMemoryData* shm, const SharedDataset* ds, int sample_idx, int proc_id, int num_procs);
void execute_backward_pass(SharedMemoryData* shm, const SharedDataset* ds, int sample_idx, int proc_id, int num_procs, float learning_rate);
