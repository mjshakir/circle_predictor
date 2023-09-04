#pragma once

//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <iostream>
#include <vector>
#include <functional>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <optional>
#include <atomic>
//--------------------------------------------------------------
namespace Utils{
    //--------------------------------------------------------------
    class ThreadPool {
        //--------------------------------------------------------------
        public:
            //--------------------------------------------------------------
            ThreadPool(void) = delete;
            //--------------------------
            explicit ThreadPool(const size_t& numThreads = std::thread::hardware_concurrency());
            //--------------------------
            ThreadPool(const ThreadPool&)            = delete;
            ThreadPool& operator=(const ThreadPool&) = delete;
            //----------------------------
            ThreadPool(ThreadPool&&)                 = delete;
            ThreadPool& operator=(ThreadPool&&)      = delete;
            //--------------------------
            ~ThreadPool(void);
            //--------------------------
            size_t active_workers_size() const;
            //--------------------------
            size_t queued_size() const;
            //--------------------------
            template <class F, class... Args>
            auto queue(F&& f, Args&&... args, const uint8_t& retries = 0) -> std::optional<std::future<std::invoke_result_t<F, Args...>>>;
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            template <class F, class... Args>
            auto enqueue(F&& f, Args&&... args, const uint8_t& retries) -> std::optional<std::future<std::invoke_result_t<F, Args...>>>;
            //--------------------------
            void create_task(const size_t& numThreads);
            //--------------------------
            void workerFunction(const std::stop_token& stoken);
            //--------------------------
            void stop(void);
            //--------------------------
            size_t activeWorkers() const;
            //--------------------------
            size_t queuedTasks() const;
            //--------------------------------------------------------------
        private:
            //--------------------------------------------------------------
            std::atomic<bool> m_stop;
            //--------------------------
            std::atomic<size_t> m_failedTasksCount, m_retriedTasksCount, m_completedTasksCount;
            //--------------------------
            struct Task {
                //--------------------------
                public:
                    //--------------------------
                    Task(void) = default;
                    //--------------------------
                    Task(const std::function<void()>& func_, const uint8_t& retries = 0)
                        : func(func_), retriesLeft(retries) {}
                    //--------------------------
                    Task(const Task&)            = default;
                    Task& operator=(const Task&) = default;
                    //----------------------------
                    Task(Task&&)                 = default;
                    Task& operator=(Task&&)      = default;
                    //--------------------------
                    std::function<void()> func;
                    uint8_t retriesLeft;
                    //--------------------------
            };
            //--------------------------
            std::vector<std::jthread> m_workers;
            //--------------------------
            std::deque<Task> m_tasks, m_lowPriorityTasks;
            //--------------------------
            std::mutex m_queueMutex;
            //--------------------------
            std::condition_variable m_taskAvailableCondition;            
        //--------------------------------------------------------------
    };// end class ThreadPool
    //--------------------------------------------------------------
}//end namespace Utils
//--------------------------------------------------------------