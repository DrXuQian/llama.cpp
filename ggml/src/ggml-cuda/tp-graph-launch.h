#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

// The owner serializes launch calls and keeps callback data alive until destruction.
class ggml_cuda_tp_graph_launcher {
public:
    using callback = void (*)(void *, size_t);

    ggml_cuda_tp_graph_launcher(size_t ranks, void * data, callback initialize, callback submit) :
        ranks(ranks), data(data), initialize(initialize), submit(submit) {
        workers.reserve(ranks);
        try {
            for (size_t rank = 0; rank < ranks; ++rank) {
                workers.emplace_back(&ggml_cuda_tp_graph_launcher::worker, this, rank);
            }
        } catch (...) {
            stop();
            throw;
        }
        std::unique_lock<std::mutex> lock(mutex);
        ready_cv.wait(lock, [&] { return initialized == ranks; });
    }

    ~ggml_cuda_tp_graph_launcher() { stop(); }

    ggml_cuda_tp_graph_launcher(const ggml_cuda_tp_graph_launcher &) = delete;
    ggml_cuda_tp_graph_launcher & operator=(const ggml_cuda_tp_graph_launcher &) = delete;

    void launch() {
        std::unique_lock<std::mutex> lock(mutex);
        ready = completed = 0;
        ++generation;
        work_cv.notify_all();
        done_cv.wait(lock, [&] { return completed == ranks; });
    }

private:
    void worker(size_t rank) {
        initialize(data, rank);
        std::unique_lock<std::mutex> lock(mutex);
        ++initialized;
        ready_cv.notify_one();
        uint64_t seen = 0;
        for (;;) {
            work_cv.wait(lock, [&] { return stopping || generation != seen; });
            if (stopping) {
                return;
            }
            seen = generation;
            if (++ready == ranks) {
                // The last ready rank releases all submissions without waking the caller.
                released.store(seen, std::memory_order_release);
            }
            lock.unlock();
            while (released.load(std::memory_order_acquire) != seen) {
                std::this_thread::yield();
            }
            // No shared lock may cover the driver call: it can wait for another rank.
            submit(data, rank);
            lock.lock();
            if (++completed == ranks) {
                done_cv.notify_one();
            }
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        work_cv.notify_all();
        for (auto & worker : workers) {
            worker.join();
        }
    }

    const size_t ranks;
    void * const data;
    const callback initialize;
    const callback submit;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable work_cv, ready_cv, done_cv;
    std::atomic<uint64_t> released{0};
    uint64_t generation = 0;
    size_t initialized = 0, ready = 0, completed = 0;
    bool stopping = false;
};
