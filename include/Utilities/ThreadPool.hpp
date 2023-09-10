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
                LOW = 0, LOW_MEDIUM, LOW_HIGH,
                MEDIUM_LOW, MEDIUM, HIGH_MEDIUM,
                HIGH_LOW, MEDIUM_HIGH, HIGH,
                CRITICAL
                //--------------------------
            }; // end enum class Priority
            //--------------------------------------------------------------
        private:
            //--------------------------------------------------------------
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
                        Task(std::function<void()> t, std::variant<Priority, uint8_t> p, uint8_t r)
                            : task(std::move(t)), priority(p), retries(r) {}
                        //--------------------------
                        std::function<void()> task;
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
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            class TaskBuilder {
                public:
                    TaskBuilder(ThreadPool& threadPool, std::function<void()> task)
                        : m_threadPool(threadPool), m_task(std::move(task)), m_priority(Priority::LOW), m_retries(0){}
                    //--------------------------
                    ~TaskBuilder(void) { 
                        m_threadPool.addTask(submit()); 
                    }
                    //--------------------------
                    TaskBuilder& set_priority(const Priority& p) { 
                        m_priority = p; 
                        return *this;
                    }
                    //--------------------------
                    TaskBuilder& set_priority(const uint8_t& p) { 
                        m_priority = p; 
                        return *this; 
                    }
                    //--------------------------
                    TaskBuilder& set_retries(const uint8_t& r) { 
                        m_retries = r; 
                        return *this; 
                    }
                    //--------------------------
                    Task submit() const { 
                        return Task(std::move(m_task), m_priority, m_retries); 
                    }
                    //--------------------------------------------------------------
                private:
                    //--------------------------
                    ThreadPool& m_threadPool;
                    std::function<void()> m_task;
                    std::variant<Priority, uint8_t> m_priority;
                    uint8_t m_retries;
                //--------------------------------------------------------------
            };// end class TaskBuilder
            //--------------------------------------------------------------
        public:
            //--------------------------------------------------------------
            ThreadPool(void) = delete;
            //--------------------------
            explicit ThreadPool(const size_t& numThreads = static_cast<size_t>(std::thread::hardware_concurrency()),
                                const size_t& minLimit = 1,
                                const size_t& maxLimit = static_cast<size_t>(std::thread::hardware_concurrency()));
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
            TaskBuilder queue(F&& f, Args&&... args){
                //--------------------------
                return enqueue(std::forward<F>(f), std::forward<Args>(args)...);
                //--------------------------
            }// end TaskBuilder queue(F&& f, Args&&... args)         
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            template <class F, class... Args>
            TaskBuilder enqueue(F&& f, Args&&... args){
                //--------------------------
                auto task = [f = std::forward<F>(f), ...args = std::forward<Args>(args)]() mutable { return f(args...); };
                return TaskBuilder(*this, [task = std::move(task)](){ task(); });
                //--------------------------
            }// end TaskBuilder enqueue(F&& f, Args&&... args)
            //--------------------------
            
            //--------------------------------------------------------------
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
            void addTask(Task&& task);
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