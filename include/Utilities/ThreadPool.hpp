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
#include <tuple>
//--------------------------------------------------------------
namespace Utils{
    //--------------------------------------------------------------
    class ThreadPool {
        //--------------------------------------------------------------
        public:
            //--------------------------------------------------------------
            enum class Priority : uint8_t {
                //--------------------------
                LOW = 0,
                LOW_MEDIUM,
                LOW_HIGH,
                MEDIUM_LOW,
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
            size_t active_workers_size(void) const;
            //--------------------------
            size_t queued_size(void) const;
            //--------------------------
            std::tuple<size_t, size_t, size_t> status(void);
            //--------------------------
            void status_disply(void);
            //--------------------------
            template <class F, class... Args>
            auto queue(F&& f, Args&&... args, const Priority& priority = Priority::MEDIUM, const uint8_t& retries = 0) 
                            -> std::optional<std::future<std::invoke_result_t<F, Args...>>>;
            //--------------------------
             template <class F, class... Args>
            auto queue(F&& f, Args&&... args, const uint8_t& priority = 100U, const uint8_t& retries = 0) 
                            -> std::optional<std::future<std::invoke_result_t<F, Args...>>>;            
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            template <class F, class... Args>
            auto enqueue(F&& f, Args&&... args, const std::variant<Priority, uint8_t>& priority, const uint8_t& retries) 
                            -> std::optional<std::future<std::invoke_result_t<F, Args...>>>;
            //--------------------------
            void create_task(const size_t& numThreads);
            //--------------------------
            void workerFunction(const std::stop_token& stoken);
            //--------------------------
            void stop(void);
            //--------------------------
            void adjustWorkers(void);
            //--------------------------
            // void adjustmentThreadFunction(void);
            //--------------------------
            void adjustmentThreadFunction(const std::stop_token& stoken);
            //--------------------------
            size_t activeWorkers(void) const;
            //--------------------------
            size_t queuedTasks(void) const;
            //--------------------------
            std::tuple<size_t, size_t, size_t> get_status(void);
            //--------------------------
            void status_display_internal(void);
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
            struct Comparator;
            struct Task;
            //--------------------------
            std::vector<std::jthread> m_workers;
            //--------------------------
            std::priority_queue<std::shared_ptr<Task>, std::vector<std::shared_ptr<Task>>> m_tasks;
            //--------------------------
            std::jthread m_adjustmentThread;
            //--------------------------
            std::mutex m_mutex;
            //--------------------------
            std::condition_variable m_taskAvailableCondition;            
        //--------------------------------------------------------------
    };// end class ThreadPool
    //--------------------------------------------------------------
}//end namespace Utils
//--------------------------------------------------------------