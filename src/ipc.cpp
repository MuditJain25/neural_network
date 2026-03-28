#include "ipc.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

SharedMemoryData* init_shared_memory(const std::string& name, int num_workers, bool is_master) {
    int shm_fd;
    if (is_master) {
        // Master creates & initializes
        shm_unlink(name.c_str()); // Remove existing if any
        shm_fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            throw std::runtime_error("shm_open failed");
        }
        if (ftruncate(shm_fd, sizeof(SharedMemoryData)) == -1) {
            throw std::runtime_error("ftruncate failed");
        }
    } else {
        // Worker opens existing
        shm_fd = shm_open(name.c_str(), O_RDWR, 0666);
        if (shm_fd == -1) {
            throw std::runtime_error("worker shm_open failed");
        }
    }

    void* ptr = mmap(0, sizeof(SharedMemoryData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap failed");
    }

    SharedMemoryData* shm = static_cast<SharedMemoryData*>(ptr);

    if (is_master) {
        // Zero all memory out to get clean accumulators
        memset(shm, 0, sizeof(SharedMemoryData));

        // Initialize Shared Mutex
        pthread_mutexattr_t m_attr;
        pthread_mutexattr_init(&m_attr);
        pthread_mutexattr_setpshared(&m_attr, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&shm->mutex, &m_attr);

        // Initialize Shared Barrier
        pthread_barrierattr_t b_attr;
        pthread_barrierattr_init(&b_attr);
        pthread_barrierattr_setpshared(&b_attr, PTHREAD_PROCESS_SHARED);
        pthread_barrier_init(&shm->barrier, &b_attr, num_workers);

        pthread_mutexattr_destroy(&m_attr);
        pthread_barrierattr_destroy(&b_attr);
    }

    return shm;
}

void cleanup_shared_memory(const std::string& name, SharedMemoryData* shm_ptr) {
    if (shm_ptr != nullptr) {
        // Cleanup the synchronisation primitives
        pthread_mutex_destroy(&shm_ptr->mutex);
        pthread_barrier_destroy(&shm_ptr->barrier);
        
        munmap(shm_ptr, sizeof(SharedMemoryData));
        shm_unlink(name.c_str());
    }
}
