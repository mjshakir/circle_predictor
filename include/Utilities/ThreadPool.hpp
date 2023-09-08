#pragma once

//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <iostream>
#include <vector>
#include <functional>
#include <queue>
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
            enum class Priority {
                //--------------------------
                LOW = 0,
                LOW_MEDIUM,
                MEDIUM,
                HIGH_MEDIUM,
                LOW_HIGH,
                MEDIUM_HIGH,
                HIGH,
                CRITICAL
                //--------------------------
            }; // end enum class Priority
            //--------------------------
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
            auto queue(F&& f, Args&&... args, const Priority& priority = Priority::LOW, const uint8_t& retries = 0) -> std::optional<std::future<std::invoke_result_t<F, Args...>>>;            
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            template <class F, class... Args>
            auto enqueue(F&& f, Args&&... args, const Priority& priority, const uint8_t& retries) -> std::optional<std::future<std::invoke_result_t<F, Args...>>>;
            //--------------------------
            void create_task(const size_t& numThreads);
            //--------------------------
            void workerFunction(const std::stop_token& stoken);
            //--------------------------
            void stop(void);
            //--------------------------
            void adjustWorkers(void);
            //--------------------------
            void adjustmentThreadFunction(void);
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
            const size_t m_lowerThreshold, m_upperThreshold;  // Example value
            bool m_adjustmentFlag;
            //--------------------------
            struct Task {
                //--------------------------
                public:
                    //--------------------------
                    Task(void)                   = default;
                    //--------------------------
                    Task(const Task&)            = default;
                    Task& operator=(const Task&) = default;
                    //----------------------------
                    Task(Task&&)                 = default;
                    Task& operator=(Task&&)      = default;
                    //--------------------------
                    std::unique_ptr<std::packaged_task<void()>> task;
                    uint8_t retries;
                    Priority priority;
                    //--------------------------
                    // Comparison for priority
                    //--------------------------
                    bool operator<(const Task& other) const {
                        //--------------------------
                        if (priority != other.priority) {
                            //--------------------------
                            return other.priority < priority;
                            //--------------------------
                        }// end if (priority != other.priority)
                        //--------------------------
                        return other.retries < retries;
                        //--------------------------
                    }// end bool operator<(const Task& other) const
                    //--------------------------
            }; // end struct Task
            //--------------------------
            std::vector<std::jthread> m_workers;
            //--------------------------
            std::priority_queue<Task> m_tasks;
            //--------------------------
            std::mutex m_mutex;
            //--------------------------
            std::condition_variable m_taskAvailableCondition;            
        //--------------------------------------------------------------
    };// end class ThreadPool
    //--------------------------------------------------------------
}//end namespace Utils
//--------------------------------------------------------------