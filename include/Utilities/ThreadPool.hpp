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
#include <future>
#include <variant>
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
            struct Comparator{
                //--------------------------
                template <typename T, typename U>
                bool operator()(T, U) const {
                    return false; // Different types, so no meaningful comparison
                }
                //--------------------------
                bool operator()(Priority lhs, Priority rhs) const {
                    return rhs < lhs;
                }
                //--------------------------
                bool operator()(uint8_t lhs, uint8_t rhs) const {
                    return rhs < lhs;
                }
                //--------------------------
            };// end struct Comparator
            //--------------------------
            struct Task{
                //--------------------------
                public:
                    //--------------------------
                    Task(void)                   = default;
                    //--------------------------
                    Task(std::unique_ptr<std::packaged_task<void()>> t, std::variant<Priority, uint8_t> p, uint8_t r)
                    : task(std::move(t)), priority(p), retries(r) {}
                    //--------------------------
                    std::unique_ptr<std::packaged_task<void()>> task;
                    // Priority Priority;
                    std::variant<Priority, uint8_t> priority;
                    uint8_t retries;
                    //--------------------------
                    // Comparison for priority
                    //--------------------------
                    bool operator<(const Task& other) const {
                        //--------------------------
                        if (priority != other.priority) {
                            return std::visit(Comparator{}, other.priority, priority);
                        }
                        return other.retries < retries;
                        //--------------------------
                    }// end bool operator<(const Task& other) const
                    //--------------------------
            };
            //--------------------------
            std::vector<std::jthread> m_workers;
            //--------------------------
            std::priority_queue<std::shared_ptr<Task>, std::vector<std::shared_ptr<Task>>> m_tasks;
            //--------------------------
            std::jthread m_adjustmentThread;
            //--------------------------
            mutable std::mutex m_mutex;
            //--------------------------
            std::condition_variable m_taskAvailableCondition;            
        //--------------------------------------------------------------
    };// end class ThreadPool
    //--------------------------------------------------------------
}//end namespace Utils
//--------------------------------------------------------------