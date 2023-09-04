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
Utils::ThreadPool::ThreadPool(const size_t& numThreads) :   m_stop(false), 
                                                            m_failedTasksCount(0),
                                                            m_retriedTasksCount(0),
                                                            m_completedTasksCount(0) {
    //--------------------------
    auto _threads_number = std::min(numThreads, static_cast<size_t>(std::thread::hardware_concurrency()));
    create_task((_threads_number > 0) ? _threads_number : 1UL);
    //--------------------------
}// end Utils::ThreadPool::ThreadPool(const size_t& numThreads)
//--------------------------------------------------------------
Utils::ThreadPool::~ThreadPool(void) {
    //--------------------------
    stop();
    //--------------------------
}// end Utils::ThreadPool::~ThreadPool(void) 
//--------------------------------------------------------------
size_t Utils::ThreadPool::active_workers_size() const{
    //--------------------------
    return activeWorkers();
    //--------------------------
}// end size_t Utils::ThreadPool::active_workers_size() const
//--------------------------------------------------------------
size_t Utils::ThreadPool::queued_size() const{
    //--------------------------
    return queuedTasks();
    //--------------------------
}// end size_t Utils::ThreadPool::queued_size() const
//--------------------------------------------------------------
template <class F, class... Args>
auto Utils::ThreadPool::queue(F&& f, Args&&... args, const uint8_t& retries) -> std::optional<std::future<std::invoke_result_t<F, Args...>>> {
    //--------------------------
    return enqueue(std::forward<F>(f), std::forward<Args>(args)..., retries);
    //--------------------------
}// end auto Utils::ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>
//--------------------------------------------------------------
template <class F, class... Args>
auto Utils::ThreadPool::enqueue(F&& f, Args&&... args, const uint8_t& retries) -> std::optional<std::future<std::invoke_result_t<F, Args...>>> {
    //--------------------------
   using return_type = std::invoke_result_t<F, Args...>;
    //--------------------------
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        //--------------------------
        [func = std::forward<F>(f), args = std::make_tuple(std::forward<Args>(args)...)]() mutable {
            //--------------------------
            return std::apply(std::move(func), std::move(args));
            //--------------------------
        }
        //--------------------------
    );
    //--------------------------
    std::future<return_type> res = task->get_future();
    //--------------------------
    { // Begin append tasks
        //--------------------------
        std::scoped_lock lock(m_queueMutex);
        //--------------------------
        if (m_stop.load()) {
            //--------------------------
            return std::nullopt;
            //--------------------------
        }// end if (m_stop.load())
        //--------------------------
        m_tasks.emplace_back([task]() { (*task)(); }, retries);
        //--------------------------
    } // End append tasks
    //--------------------------
    m_taskAvailableCondition.notify_one();
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
void Utils::ThreadPool::workerFunction(const std::stop_token& stoken) {
    //--------------------------
    while (!stoken.stop_requested()) {
        //--------------------------
        Task task;
        //--------------------------
        { //Begin added tasks 
            //--------------------------
            std::unique_lock lock(m_queueMutex);
            //--------------------------
            m_taskAvailableCondition.wait(lock, [this, &stoken] { return stoken.stop_requested() or !m_tasks.empty(); });
            //--------------------------
            if (stoken.stop_requested() and m_tasks.empty() and m_lowPriorityTasks.empty()){
                //--------------------------
                return;
                //--------------------------
            }// end if (stoken.stop_requested() and m_tasks.empty())
            //--------------------------
            task = m_tasks.empty() ? std::move(m_lowPriorityTasks.front()) : std::move(m_tasks.front());
            //--------------------------
            if (m_tasks.empty()) {
                //--------------------------
                m_lowPriorityTasks.pop_front();
                //--------------------------
            } else {
                //--------------------------
                m_tasks.pop_front();
                //--------------------------
            }// end else
            //--------------------------
        } //End added tasks 
        //--------------------------
        try {
            //--------------------------
            task.func();
            //--------------------------
            ++m_completedTasksCount;
            //--------------------------
        } // end try
        catch (const std::exception& e) {
            //--------------------------
            if (task.retriesLeft > 0) {
                //--------------------------
                std::scoped_lock lock(m_queueMutex);
                //--------------------------
                --task.retriesLeft;
                //--------------------------
                m_lowPriorityTasks.emplace_back(std::move(task));
                //--------------------------
                ++m_retriedTasksCount;
                //--------------------------
            } else {
                //--------------------------
                ++m_failedTasksCount;
                //--------------------------
                std::cerr << "Error in task after multiple retries: " << e.what() << std::endl;
                //--------------------------
            }// end else
            //--------------------------
        }// end catch (const std::exception& e)
        catch (...) {
            //--------------------------
            ++m_failedTasksCount;
            //--------------------------
            std::cerr << "Unknown error occurred in task." << std::endl;
            //--------------------------
        }//end catch (...)
        //--------------------------
    }// end while (!stoken.stop_requested())
    //--------------------------
}// end void Utils::ThreadPool::workerFunction(void)
//--------------------------------------------------------------
void Utils::ThreadPool::stop(void){
    //--------------------------
     { // set m_stop to true
        //--------------------------
        std::scoped_lock lock(m_queueMutex);
        //--------------------------
        m_stop.store(true);
        //--------------------------
    }// end set
    //--------------------------
    m_taskAvailableCondition.notify_all();
    //--------------------------
    for (auto &worker : m_workers) {
        //--------------------------
        worker.request_stop();
        //--------------------------
    }// end for (auto &worker : m_workers)
    //--------------------------
    m_taskAvailableCondition.notify_all();
    //--------------------------
}//end void Utils::ThreadPool::stop(void)
//--------------------------------------------------------------
size_t Utils::ThreadPool::activeWorkers() const{
    //--------------------------
    return m_workers.size();
    //--------------------------
}// end size_t Utils::ThreadPool::activeWorkers() const
//--------------------------------------------------------------
size_t Utils::ThreadPool::queuedTasks() const{
    //--------------------------
    std::unique_lock lock(m_queueMutex);
    //--------------------------
    return m_tasks.size();
    //--------------------------
}// end size_t Utils::ThreadPool::queuedTasks() const
//--------------------------------------------------------------