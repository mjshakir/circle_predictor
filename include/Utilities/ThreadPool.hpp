#pragma once

//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <iostream>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <tuple>
#include <future>
#include <variant>
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Utilities/PriorityQueue.hpp"
#include "Utilities/ThreadTask.hpp"
//--------------------------------------------------------------
namespace Utils{
    //--------------------------------------------------------------
    class ThreadPool {
        //-------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            template <typename F, typename... Args>
            class TaskBuilder {
                public:
                    //--------------------------------------------------------------
                    TaskBuilder(ThreadPool& threadPool, F&& f, Args&&... args)
                        :   m_threadPool(threadPool),
                            m_task([f = std::forward<F>(f), ...args = std::forward<Args>(args)]() mutable {
                                return f(args...);
                            }),
                            m_priority(0),
                            m_retries(0) {
                            m_future = m_task.get_future();
                    }// end TaskBuilder(ThreadPool& threadPool, F&& f, Args&&... args)
                    //--------------------------
                    ~TaskBuilder(void) {
                        m_threadPool.addTask(std::move(m_task), m_priority, m_retries);
                    }// end  ~TaskBuilder(void)
                    //--------------------------
                    TaskBuilder& set_priority(const uint8_t& p) {
                        m_priority = p;
                        return *this;
                    }// end TaskBuilder& set_priority(const uint8_t& p)
                    //--------------------------
                    TaskBuilder& set_retries(const uint8_t& r) {
                        m_retries = r;
                        return *this;
                    }// end TaskBuilder& set_retries(const uint8_t& r)
                    //--------------------------
                    template <typename T>
                    operator std::future<T>() { return std::move(m_future); }
                    //--------------------------------------------------------------
                private:
                    //--------------------------------------------------------------
                    ThreadPool& m_threadPool;
                    std::packaged_task<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>()> m_task;
                    std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>> m_future;
                    uint8_t m_priority, m_retries;
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
            auto queue(F&& f, Args&&... args){
                //--------------------------
                return enqueue(std::forward<F>(f), std::forward<Args>(args)...);
                //--------------------------
            }// end TaskBuilder queue(F&& f, Args&&... args)         
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            template <class F, class... Args>
            auto enqueue(F&& f, Args&&... args){
                //--------------------------
                return TaskBuilder(*this, std::forward<F>(f), std::forward<Args>(args)...);
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
            void addTask(Utils::ThreadTask&& task);
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
            Utils::PriorityQueue<Utils::ThreadTask> m_tasks;
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