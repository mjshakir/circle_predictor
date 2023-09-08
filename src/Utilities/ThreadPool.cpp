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
#include <chrono>
//--------------------------------------------------------------
// Definitions
//--------------------------
constexpr size_t SOME_MAX_LIMIT = 100; // Example
constexpr size_t SOME_MIN_LIMIT = 1;   // Example
constexpr auto CHECK_INTERVAL   = std::chrono::milliseconds(10);  // For example, check every 10 seconds
//--------------------------------------------------------------
Utils::ThreadPool::ThreadPool(const size_t& numThreads) :   m_stop(false), 
                                                            m_failedTasksCount(0),
                                                            m_retriedTasksCount(0),
                                                            m_completedTasksCount(0),
                                                            m_lowerThreshold(1UL),
                                                            m_upperThreshold(static_cast<size_t>(std::thread::hardware_concurrency())){
    //--------------------------
    auto _threads_number = std::min(numThreads, m_upperThreshold);
    create_task((_threads_number > 0) ? _threads_number : m_lowerThreshold);
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
auto Utils::ThreadPool::queue(F&& f, Args&&... args, const Priority& priority, const uint8_t& retries) -> std::optional<std::future<std::invoke_result_t<F, Args...>>> {
    //--------------------------
    return enqueue(std::forward<F>(f), std::forward<Args>(args)..., priority, retries);
    //--------------------------
}// end auto Utils::ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>
//--------------------------------------------------------------
template <class F, class... Args>
auto Utils::ThreadPool::enqueue(F&& f, Args&&... args, const Priority& priority, const uint8_t& retries) -> std::optional<std::future<std::invoke_result_t<F, Args...>>> {
    //--------------------------
   using return_type = std::invoke_result_t<F, Args...>;
    //--------------------------
    auto task = std::make_unique<std::packaged_task<return_type()>>(
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
        std::scoped_lock lock(m_mutex);
        //--------------------------
        if (m_stop.load()) {
            //--------------------------
            return std::nullopt;
            //--------------------------
        }// end if (m_stop.load())
        //--------------------------
        m_tasks.emplace(Task{std::move(task), priority, retries});
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
        // m_workers.emplace_back(&ThreadPool::workerFunction, this);
        //--------------------------
        m_workers.emplace_back(&ThreadPool::adjustmentThreadFunction, this);
        //--------------------------
    }// end  for (size_t i = 0; i < numThreads; ++i)
    //--------------------------
}//end void Utils::ThreadPool::create_task(const size_t& numThreads)
//--------------------------------------------------------------
void Utils::ThreadPool::workerFunction(const std::stop_token& stoken) {
    //--------------------------
    while (!stoken.stop_requested()) {
        //--------------------------
            std::unique_lock lock(m_mutex);
            //--------------------------
            m_taskAvailableCondition.wait(lock, [this, &stoken] {return stoken.stop_requested() or !m_tasks.empty();});
            //--------------------------
            if (stoken.stop_requested() and m_tasks.empty() and m_stop.load()) {
                //--------------------------
                return;
                //--------------------------
            }// end if (stoken.stop_requested() && m_tasks.empty())
            //--------------------------
            auto task = m_tasks.top(); // Make a copy of the task
            //--------------------------
            m_tasks.pop();
            //--------------------------
        try {
            //--------------------------
            if (task.task) {
                //--------------------------
                (*task.task)();
                //--------------------------
                ++m_completedTasksCount;
                //--------------------------
            }// end if (task.task)
            //--------------------------
        } // end try
        catch (const std::exception& e) {
            //--------------------------
            if (task.retries > 0) {
                //--------------------------
                std::scoped_lock lock(m_mutex);
                //--------------------------
                --task.retries;
                //--------------------------
                m_tasks.push(task); // Put the modified copy of the task back in the queue
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
        } // end catch (const std::exception& e)
        catch (...) {
            //--------------------------
            ++m_failedTasksCount;
            //--------------------------
            std::cerr << "Unknown error occurred in task." << std::endl;
            //--------------------------
        }// end catch (...)
        //--------------------------
    }// end while (!stoken.stop_requested())
    //--------------------------
}// end void Utils::ThreadPool::workerFunction(void)
//--------------------------------------------------------------
void Utils::ThreadPool::stop(void){
    //--------------------------
     { // set m_stop to true
        //--------------------------
        std::scoped_lock lock(m_mutex);
        m_stop.store(true);
        //--------------------------
    }// end set
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
void Utils::ThreadPool::adjustWorkers(void) {
    //--------------------------
    std::scoped_lock lock(m_mutex);
    //--------------------------
    auto taskCount      = queuedTasks();
    auto workerCount    = activeWorkers();
    //--------------------------
    if (taskCount > m_upperThreshold and workerCount < SOME_MAX_LIMIT) {
        //--------------------------
        // Determine the number of threads to create based on the tasks and the remaining available slots
        //--------------------------
       auto threadsToCreate = std::min(std::max(static_cast<size_t>(0), taskCount - m_upperThreshold),
                                        SOME_MAX_LIMIT - workerCount);

        threadsToCreate = std::min(threadsToCreate, 
                            std::max(static_cast<size_t>(0), std::thread::hardware_concurrency() - workerCount));
        //--------------------------
        create_task(threadsToCreate);
        //--------------------------
    }// end  if (taskCount > m_upperThreshold and workerCount < SOME_MAX_LIMIT)
    //--------------------------
    if (taskCount < m_lowerThreshold and workerCount > SOME_MIN_LIMIT) {
        //--------------------------
        if (workerCount > 1) {
            //--------------------------
            m_workers.back().request_stop();
            m_workers.pop_back();
            //--------------------------
        }// end if (workerCount > 1)
    }// end if (taskCount < m_lowerThreshold && workerCount > SOME_MIN_LIMIT)
    //--------------------------
}// end void Utils::ThreadPool::adjustWorkers(void)
//--------------------------------------------------------------
void Utils::ThreadPool::adjustmentThreadFunction(void) {
    //--------------------------
    while (!m_stop.load()) {
        //--------------------------
        std::unique_lock<std::mutex> lock(m_mutex);
        //--------------------------
        m_taskAvailableCondition.wait_for(lock, CHECK_INTERVAL, [this] { return m_stop.load(); });
        //--------------------------
        if (m_stop.load()){
            //--------------------------
            return;
            //--------------------------
        }// end if (m_stop.load())
        //--------------------------
        adjustWorkers();
        //--------------------------
    }// end while (!m_stop.load())
    //--------------------------
}// end void Utils::ThreadPool::adjustmentThreadFunction(void)
//--------------------------------------------------------------
size_t Utils::ThreadPool::activeWorkers() const{
    //--------------------------
    return m_workers.size();
    //--------------------------
}// end size_t Utils::ThreadPool::activeWorkers() const
//--------------------------------------------------------------
size_t Utils::ThreadPool::queuedTasks() const{
    //--------------------------
    std::unique_lock lock(m_mutex);
    //--------------------------
    return m_tasks.size();
    //--------------------------
}// end size_t Utils::ThreadPool::queuedTasks() const
//--------------------------------------------------------------