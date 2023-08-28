//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Utilities/ThreadPool.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <future>       
#include <type_traits>
#include <memory>
#include <algorithm>
//--------------------------------------------------------------
Utils::ThreadPool::ThreadPool(void) : m_stop(false){
    //--------------------------
    create_task(std::thread::hardware_concurrency());
    //--------------------------
}// end Utils::ThreadPool::ThreadPool(void) 
//--------------------------------------------------------------
Utils::ThreadPool::ThreadPool(const size_t& numThreads) : m_stop(false) {
    //--------------------------
    create_task(std::min(numThreads, static_cast<size_t>(std::thread::hardware_concurrency())));
    //--------------------------
}// end Utils::ThreadPool::ThreadPool(const size_t& numThreads)
//--------------------------------------------------------------
Utils::ThreadPool::~ThreadPool(void) {
    //--------------------------
    { // set m_stop to true
        //--------------------------
        std::unique_lock<std::mutex> lock(m_queueMutex);
        //--------------------------
        m_stop = true;
        //--------------------------
    }// end set
    //--------------------------
    m_condition.notify_all();
    //--------------------------
    for (auto &worker : m_workers) {
        //--------------------------
        worker.join();
        //--------------------------
    }// end for (auto &worker : m_workers)
    //--------------------------
}// end Utils::ThreadPool::~ThreadPool(void) 
//--------------------------------------------------------------
template <class F, class... Args>
auto Utils::ThreadPool::queue(F&& f, Args&&... args) -> std::optional<std::future<std::invoke_result_t<F, Args...>>> {
    //--------------------------
    return enqueue(std::forward<F>(f), std::forward<Args>(args)...);
    //--------------------------
}// end auto Utils::ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>
//--------------------------------------------------------------
template <class F, class... Args>
auto Utils::ThreadPool::enqueue(F&& f, Args&&... args) -> std::optional<std::future<std::invoke_result_t<F, Args...>>> {
    //--------------------------
    using return_type = std::invoke_result_t<F, Args...>;
    //--------------------------
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        //--------------------------
        [func = std::forward<F>(f), args = std::make_tuple(std::forward<Args>(args)...)]() mutable {
            //--------------------------
            std::apply(std::move(func), std::move(args));
            //--------------------------
        }
        //--------------------------
    );
    //--------------------------
    std::future<return_type> res = task->get_future();
    //--------------------------
    { // begin append tasks 
        //--------------------------
        std::unique_lock<std::mutex> lock(m_queueMutex);
        //--------------------------
        if (m_stop) {
            //--------------------------
            return std::nullopt;
            //--------------------------
        }// end if (m_stop)
        //--------------------------
        m_tasks.emplace_back([task]() { (*task)(); });
        //--------------------------
    } // end append tasks 
    //--------------------------
    m_condition.notify_one();
    //--------------------------
    return res;
    //--------------------------
}// end auto Utils::ThreadPool::enqueue_local(F&& f, Args&&... args) -> std::optional<std::future<std::invoke_result_t<F, Args...>>>
//--------------------------------------------------------------
void Utils::ThreadPool::create_task(const size_t& numThreads){
    //--------------------------
    m_workers.reserve(numThreads);
    //--------------------------
    for (size_t i = 0; i < numThreads; ++i) {
        //--------------------------
        m_workers.emplace_back(&ThreadPool::workerFunction, this);
        //--------------------------
    }// end  for (size_t i = 0; i < numThreads; ++i)
    //--------------------------
}//end void Utils::ThreadPool::create_task(const size_t& numThreads)
//--------------------------------------------------------------
void Utils::ThreadPool::workerFunction(void) {
    //--------------------------
    while (true) {
        //--------------------------
        std::function<void()> task;
        //--------------------------
        { // begin append and pop m_tasks 
            //--------------------------
            std::scoped_lock lock(m_queueMutex);
            //--------------------------
            m_condition.wait(lock, [this] { return m_stop or !m_tasks.empty(); });
            //--------------------------
            if (m_stop and m_tasks.empty()) {
                //--------------------------
                return;
                //--------------------------
            }// end if (m_stop and m_tasks.empty())
            //--------------------------
            task = std::move(m_tasks.front());
            //--------------------------
            m_tasks.pop_front();
            //--------------------------
        } // end append and pop m_tasks
        //--------------------------
        task();
        //--------------------------
    }// end while (true)
    //--------------------------
}// end void Utils::ThreadPool::workerFunction(void)
//--------------------------------------------------------------