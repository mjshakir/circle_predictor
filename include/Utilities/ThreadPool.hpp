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
                //--------------------------------------------------------------
                private:
                    using ReturnType = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
                    //--------------------------
                    // Dummy struct for void-returning functions
                    struct VoidType {};
                    //--------------------------------------------------------------
                public:
                    TaskBuilder(ThreadPool& threadPool, F&& f, Args&&... args)
                        :   m_threadPool(threadPool),
                            m_priority(0),
                            m_retries(0),
                            m_submitted(false),
                            m_task(createTask(std::forward<F>(f), std::forward<Args>(args)...)) {

                        if constexpr (!std::is_void_v<ReturnType>) {
                            m_future.emplace(m_task.get_future());
                        }
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
                    void submit(void) {
                        if (!m_submitted) {
                            m_threadPool.addTask(std::move(m_task), m_priority, m_retries);
                            m_submitted = true;
                        }
                    }
                    //--------------------------
                    // Specialization for non-void tasks
                    template <typename T = ReturnType>
                    std::enable_if_t<!std::is_void_v<T>, T> get(void) {
                        auto res = m_future->get();
                        m_future.reset();  // Prevent future gets
                        return res;
                    }// end std::enable_if_t<!std::is_void_v<T>, T> get(void)
                    //--------------------------
                    // Specialization for void tasks
                    template <typename T = ReturnType>
                    std::enable_if_t<std::is_void_v<T>> get(void) {
                        static_assert(std::is_void_v<T>, "This get() should only be instantiated for void tasks.");
                    }// end std::enable_if_t<std::is_void_v<T>> get(void) 
                    //--------------------------------------------------------------
                private:
                    template <typename Func, typename... CArgs>
                    auto createTask(Func&& func, CArgs&&... capturedArgs) {
                        if constexpr (!std::is_void_v<ReturnType>) {
                            return std::packaged_task<ReturnType()>(
                                [f = std::forward<Func>(func), ...args = std::forward<CArgs>(capturedArgs)]() mutable {
                                    return std::invoke(f, std::forward<CArgs>(args)...);
                                }
                            );
                        } else {
                            return std::packaged_task<VoidType()>(
                                [f = std::forward<Func>(func), ...args = std::forward<CArgs>(capturedArgs)]() mutable {
                                    std::invoke(f, std::forward<CArgs>(args)...);
                                    return VoidType{};
                                }
                            );
                        }
                    }// end auto createTask(Func&& func, CArgs&&... capturedArgs)
                    //--------------------------------------------------------------
                    ThreadPool& m_threadPool;
                    uint8_t m_priority, m_retries;
                    bool m_submitted;   // Track if task has been submitted
                    std::packaged_task<std::conditional_t<std::is_void_v<ReturnType>, VoidType, ReturnType>()> m_task;
                    std::optional<std::future<ReturnType>> m_future;
                //--------------------------------------------------------------
            };// end class TaskBuilder
            //--------------------------------------------------------------
        public:
            //--------------------------------------------------------------
            ThreadPool(void) = delete;
            //--------------------------
            explicit ThreadPool(const size_t& numThreads = static_cast<size_t>(std::thread::hardware_concurrency()));
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
                return enqueue<F, Args...>(std::forward<F>(f), std::forward<Args>(args)...);
                //--------------------------
            }// end TaskBuilder queue(F&& f, Args&&... args)         
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            template <class F, class... Args>
            auto enqueue(F&& f, Args&&... args) {
                //--------------------------
                return TaskBuilder<F, Args...>(*this, std::forward<F>(f), std::forward<Args>(args)...);
                //--------------------------
            }// end TaskBuilder enqueue(F&& f, Args&&... args)            
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
            template<typename... Args>
            void addTask(Args&&... args){
                //--------------------------
                std::scoped_lock lock(m_mutex);
                //--------------------------
                m_tasks.emplace(std::forward<Args>(args)...);
                //--------------------------
                m_taskAvailableCondition.notify_one();
                //--------------------------
            }// end void addTask(Args&&... args)
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
            const size_t m_lowerThreshold, m_upperThreshold;
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