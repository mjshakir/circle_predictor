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
//--------------------------------------------------------------
namespace Utils{
    //--------------------------------------------------------------
    class ThreadPool {
        //--------------------------------------------------------------
        public:
            //--------------------------------------------------------------
            ThreadPool(void);
            //--------------------------
            explicit ThreadPool(const size_t& numThreads);
            //--------------------------
            ~ThreadPool(void);
            //--------------------------
            template <class F, class... Args>
            auto enqueue(F&& f, Args&&... args) -> std::optional<std::future<std::invoke_result_t<F, Args...>>>;
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            template <class F, class... Args>
            auto enqueue_local(F&& f, Args&&... args) -> std::optional<std::future<std::invoke_result_t<F, Args...>>>;
            //--------------------------
            void create_task(const size_t& numThreads);
            //--------------------------
            void workerFunction(void);
            //--------------------------------------------------------------
        private:
            //--------------------------------------------------------------
            bool m_stop;
            //--------------------------
            std::vector<std::jthread> m_workers;
            //--------------------------
            std::deque<std::function<void()>> m_tasks;
            //--------------------------
            std::mutex m_queueMutex;
            //--------------------------
            std::condition_variable m_condition;
        //--------------------------------------------------------------
    };// end class ThreadPool
    //--------------------------------------------------------------
}//end namespace Utils
//--------------------------------------------------------------