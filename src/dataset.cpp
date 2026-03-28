#include "dataset.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Reverse integer bytes for MNIST format
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

SharedDataset init_shared_dataset(const std::string& shm_name, int max_samples, bool is_master) {
    size_t images_sz = max_samples * MNIST_IMAGE_SIZE;
    size_t labels_sz = max_samples;
    size_t total_sz = images_sz + labels_sz;

    int shm_fd;
    if (is_master) {
        shm_unlink(shm_name.c_str());
        shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) throw std::runtime_error("dataset shm_open failed");
        if (ftruncate(shm_fd, total_sz) == -1) throw std::runtime_error("dataset ftruncate failed");
    } else {
        shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
        if (shm_fd == -1) throw std::runtime_error("dataset worker shm_open failed");
    }

    void* ptr = mmap(0, total_sz, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) throw std::runtime_error("dataset mmap failed");

    SharedDataset ds;
    ds.num_samples = max_samples;
    ds.images = static_cast<uint8_t*>(ptr);
    ds.labels = ds.images + images_sz;

    return ds;
}

void load_mnist_multithreaded(const std::string& image_path, const std::string& label_path, SharedDataset& dataset, int num_threads) {
    // Read headers first to ensure validity
    std::ifstream img_file(image_path, std::ios::binary);
    std::ifstream lbl_file(label_path, std::ios::binary);

    if (!img_file.is_open() || !lbl_file.is_open()) {
        throw std::runtime_error("Failed to open MNIST files");
    }

    uint32_t magic_img, num_imgs, rows, cols;
    img_file.read(reinterpret_cast<char*>(&magic_img), 4);
    img_file.read(reinterpret_cast<char*>(&num_imgs), 4);
    img_file.read(reinterpret_cast<char*>(&rows), 4);
    img_file.read(reinterpret_cast<char*>(&cols), 4);

    uint32_t magic_lbl, num_lbls;
    lbl_file.read(reinterpret_cast<char*>(&magic_lbl), 4);
    lbl_file.read(reinterpret_cast<char*>(&num_lbls), 4);

    int total_items = swap_endian(num_imgs);
    if (total_items > dataset.num_samples) total_items = dataset.num_samples;
    dataset.num_samples = total_items; // override with actual read

    img_file.close();
    lbl_file.close();

    // Define thread loading function
    auto load_chunk = [&](int start_idx, int end_idx) {
        std::ifstream img(image_path, std::ios::binary);
        std::ifstream lbl(label_path, std::ios::binary);

        img.seekg(16 + start_idx * MNIST_IMAGE_SIZE, std::ios::beg);
        lbl.seekg(8 + start_idx, std::ios::beg);

        img.read(reinterpret_cast<char*>(dataset.images + start_idx * MNIST_IMAGE_SIZE), (end_idx - start_idx) * MNIST_IMAGE_SIZE);
        lbl.read(reinterpret_cast<char*>(dataset.labels + start_idx), (end_idx - start_idx));

        img.close();
        lbl.close();
    };

    std::vector<std::thread> workers;
    int chunk_size = total_items / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? total_items : start + chunk_size;
        workers.emplace_back(load_chunk, start, end);
    }

    for (auto& t : workers) {
        t.join();
    }
}

void cleanup_shared_dataset(const std::string& shm_name, SharedDataset& dataset) {
    size_t total_sz = (dataset.num_samples * MNIST_IMAGE_SIZE) + dataset.num_samples;
    munmap(dataset.images, total_sz);
    shm_unlink(shm_name.c_str());
}
