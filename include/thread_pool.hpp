#pragma once

#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <functional>
#include <atomic>
#include <iostream>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE             /* See feature_test_macros(7) */
#endif
#include <sched.h>

int get_cpu_affinity_size() {
    cpu_set_t cpus;
    if (sched_getaffinity(0, sizeof(cpus), &cpus) < 0) {
        perror("sched_getaffinity");
        return std::thread::hardware_concurrency();
    }
    for(int i=0; i<CPU_SETSIZE ;i++) {
        if (CPU_ISSET(i, &cpus)) {
            std::cout << "[cpu " << i << "]";
        }
    }
    std::cout << std::endl;
    return CPU_COUNT(&cpus);
}

// https://stackoverflow.com/questions/15752659/thread-pooling-in-c11
//  the main thread that queue jobs also should run part of the work
//
class ThreadPool {
public:
    uint32_t num_threads;
    uint32_t num_worker_threads;
    cpu_set_t cpus;
    std::map<int, int> tid2cpu;

    ~ThreadPool() {
        Stop();
    }
    void Start() {
        // the first worker thread is main thread itself
        if (sched_getaffinity(0, sizeof(cpus), &cpus) < 0) {
            perror("sched_getaffinity");
            abort();
        }
        {
            int tid = 0;
            for(int i=0; i<CPU_SETSIZE ;i++) {
                if (CPU_ISSET(i, &cpus)) {
                    tid2cpu[tid++] = i;
                }
            }
        }

        num_threads = CPU_COUNT(&cpus);
        num_worker_threads = num_threads - 1;
        for (uint32_t i = 0; i < num_worker_threads; i++) {
            nt_flags.emplace_back(0);
            threads.emplace_back(&ThreadPool::ThreadLoop, this, 1+i, num_worker_threads+1, std::ref(nt_flags.back()));
        }
        bind_cpu(tid2cpu[0]);
        std::cout << "ThreadPool with " << num_worker_threads + 1 << " worker threads is created!" << std::endl;
    }

    // job(int thread_id, int total_threads)
    void Paralell_NT(const std::function<void(int, int)>& job) {
        nt_job = job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // set flags for each worker threads
            for (uint32_t i = 0; i < num_worker_threads; i++)
                nt_flags[i].store(1);
        }

        // inform all worker threads
        mutex_condition.notify_all();
 
        // main thread as 0-th worker thread
        job(0, num_worker_threads+1);

        // busy wait to minimize wait latency
        // (only good choice when HW concurrency is used)
        //std::unique_lock<std::mutex> lock(finished_mutex);
        //mutex_finished.wait(lock, [this] {
        //    for (uint32_t i = 0; i < num_worker_threads; i++) {
        //        if (nt_flags[i].load() > 0)
        //            return false;
        //    }
        //    return true;
        //});
        for (uint32_t i = 0; i < num_worker_threads; i++) {
            while (nt_flags[i].load() > 0);
        }
    }

    void Stop() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            should_terminate = true;
        }
        mutex_condition.notify_all();
        for (std::thread& active_thread : threads) {
            active_thread.join();
        }
        threads.clear();
    }

private:
    void bind_cpu(int cpu_id) {
        cpu_set_t cpuset;
        pthread_t thread;
        thread = pthread_self();
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        int ret = pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
        if (ret != 0) {
            std::abort();
        }
    }
    void ThreadLoop(int thread_id, int total_threads, std::atomic<int>& nt_flag) {
        bind_cpu(tid2cpu[thread_id]);
        while (true) {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                mutex_condition.wait(lock, [this, &nt_flag] {
                    return should_terminate || nt_flag.load() > 0;
                });
                if (should_terminate) {
                    return;
                }
            }

            nt_job(thread_id, total_threads);
            nt_flag.store(0);

            //mutex_finished.notify_one();
        }
    }

    bool should_terminate = false;           // Tells threads to stop looking for jobs
    std::mutex queue_mutex;                  // Prevents data races to the job queue
    std::condition_variable mutex_condition; // Allows threads to wait on new jobs or termination 
    std::vector<std::thread> threads;

    std::mutex finished_mutex;
    std::condition_variable mutex_finished;

    // instead of fetch from common jobs queue, parallel NT has it's own
    // per-thread job allocation
    std::function<void(int, int)> nt_job;
    std::deque<std::atomic<int>> nt_flags;
};

template <typename T, typename Q>
inline void splitter(const T& n, const Q& team, const Q& tid, T& n_start, T& n_end) {
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_end = n;
    } else {
        T n1 = (n + (T)team - 1) / (T)team;
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_end = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}


void test_parallel_nt() {
    ThreadPool thp;
    thp.Start();

    while(1) {
        std::atomic<int> test_parallel(0);
        thp.Paralell_NT([&](int id, int cnt) {
            test_parallel++;
        });
        
        if (test_parallel != thp.num_threads) {
            std::cout << "Error" << std::endl;
        }
        else {
            std::cout << "." << std::flush;
        }
        // allow for Ctrl+C to work
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

