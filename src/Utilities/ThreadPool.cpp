//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Utilities/ThreadPool.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------      
#include <type_traits>
#include <memory>
#include <algorithm>
#include <chrono>
//--------------------------------------------------------------
// Definitions
//--------------------------
constexpr size_t MAX_LIMIT      = 100UL; // Example
constexpr size_t MIN_LIMIT      = 1UL;   // Example
constexpr auto CHECK_INTERVAL   = std::chrono::milliseconds(10);  // For example, check every 10 seconds
//--------------------------------------------------------------
Utils::ThreadPool::ThreadPool(const size_t& numThreads) :   m_stop(false), 
                                                            m_failedTasksCount(0),
                                                            m_retriedTasksCount(0),
                                                            m_completedTasksCount(0),
                                                            m_lowerThreshold(MIN_LIMIT),
                                                            m_upperThreshold(static_cast<size_t>(std::thread::hardware_concurrency())){
    //--------------------------
    auto _threads_number = std::min(numThreads, m_upperThreshold);
    create_task((_threads_number > 0) ? _threads_number : m_lowerThreshold);
    //--------------------------
    // create the adjustment thread
    m_adjustmentThread = std::jthread(&ThreadPool::adjustmentThreadFunction, this, m_adjustmentThread.get_stop_token());
    //--------------------------
}// end Utils::ThreadPool::ThreadPool(const size_t& numThreads)
//--------------------------------------------------------------
Utils::ThreadPool::~ThreadPool(void) {
    //--------------------------
    stop();
    //--------------------------
}// end Utils::ThreadPool::~ThreadPool(void) 
//--------------------------------------------------------------
size_t Utils::ThreadPool::active_workers_size(void) const{
    //--------------------------
    return activeWorkers();
    //--------------------------
}// end size_t Utils::ThreadPool::active_workers_size() const
//--------------------------------------------------------------
size_t Utils::ThreadPool::queued_size(void) const{
    //--------------------------
    return queuedTasks();
    //--------------------------
}// end size_t Utils::ThreadPool::queued_size() const
//--------------------------------------------------------------
std::tuple<size_t, size_t, size_t> Utils::ThreadPool::status(void){
    //--------------------------
    return get_status();
    //--------------------------
}// end std::tuple<size_t, size_t, size_t> Utils::ThreadPool::get_status(void)
//--------------------------------------------------------------
void Utils::ThreadPool::status_disply(void){
    //--------------------------
    status_display_internal();
    //--------------------------
}// end void Utils::ThreadPool::status_disply(void)
//--------------------------------------------------------------
template <class F, class... Args>
TaskBuilder Utils::ThreadPool::queue(F&& f, Args&&... args){
    //--------------------------
    return enqueue(std::forward<F>(f), std::forward<Args>(args)...);
    //--------------------------
}// end auto Utils::ThreadPool::enqueue(F&& f, Args&&... args, const Priority& priority, const uint8_t& retries) -> std::future<std::invoke_result_t<F, Args...>>
//--------------------------------------------------------------
template <class F, class... Args>
TaskBuilder Utils::ThreadPool::enqueue(F&& f, Args&&... args){
    //--------------------------
    auto task = [f = std::forward<F>(f), args = std::forward<Args>(args)...]() { f(args...); };
    return TaskBuilder(*this, [task = std::move(task)](){ task(); });
    //--------------------------
}// end auto auto Utils::ThreadPool::enqueue(F&& f, Args&&... args, const Priority& priority, const uint8_t& retries) -> std::optional<std::future<std::invoke_result_t<F, Args...>>>
//--------------------------------------------------------------
void Utils::ThreadPool::create_task(const size_t& numThreads){
    //--------------------------
    m_workers.reserve(numThreads);
    //--------------------------
    for (size_t i = 0; i < numThreads; ++i) {
        //--------------------------
        // m_workers.emplace_back(&ThreadPool::workerFunction, this);
        //--------------------------
        m_workers.emplace_back([this](std::stop_token stoken) {
            this->workerFunction(stoken);
        });
        //--------------------------
    }// end  for (size_t i = 0; i < numThreads; ++i)
    //--------------------------
}//end void Utils::ThreadPool::create_task(const size_t& numThreads)
//--------------------------------------------------------------
void Utils::ThreadPool::workerFunction(const std::stop_token& stoken) {
    //--------------------------
    while (!stoken.stop_requested()) {
        //--------------------------
        std::shared_ptr<Task> task;
        {// being Append tasks 
            //--------------------------
            std::unique_lock lock(m_mutex);
            //--------------------------
            m_taskAvailableCondition.wait(lock, [this, &stoken] {return stoken.stop_requested() or !m_tasks.empty();});
            //--------------------------
            if (stoken.stop_requested() and m_tasks.empty() and m_stop.load()) {
                //--------------------------
                return;
                //--------------------------
            }// end if (stoken.stop_requested() && m_tasks.empty() && m_stop.load())
            //--------------------------
            task = m_tasks.top();
            m_tasks.pop();
            //--------------------------
        }// end // being Append tasks
        try {
            //--------------------------
            task->task();
            //--------------------------
            ++m_completedTasksCount;
            //--------------------------
        } // end try
        catch (const std::exception& e) {
            //--------------------------
            if (task->retries > 0) {
                //--------------------------
                std::scoped_lock lock(m_mutex);
                //--------------------------
                --task->retries;
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
    m_taskAvailableCondition.notify_all();
    //--------------------------
    m_adjustmentThread.request_stop();
    //--------------------------
    for (auto &worker : m_workers) {
        //--------------------------
        worker.request_stop();
        //--------------------------
    }// end for (auto &worker : m_workers)
    //--------------------------
}//end void Utils::ThreadPool::stop(void)
//--------------------------------------------------------------
void Utils::ThreadPool::adjustWorkers(void) {
    //--------------------------
    std::scoped_lock lock(m_mutex);
    //--------------------------
    auto taskCount = queuedTasks();
    auto workerCount = activeWorkers();
    //--------------------------
    if (taskCount > m_upperThreshold and workerCount < MAX_LIMIT) {
        //--------------------------
        auto threadsToCreate = std::min(std::max(static_cast<size_t>(0), taskCount - m_upperThreshold),
                                        MAX_LIMIT - workerCount);
        //--------------------------
        threadsToCreate = std::min(threadsToCreate,
                                   std::max(static_cast<size_t>(0), std::thread::hardware_concurrency() - workerCount));
        //--------------------------
        create_task(threadsToCreate);
        //--------------------------
    }// end if (taskCount > m_upperThreshold and workerCount < MAX_LIMIT)
    //--------------------------
    if (taskCount < m_lowerThreshold and workerCount > MIN_LIMIT) {
        //--------------------------
        if (workerCount > 1) {
            //--------------------------
            m_workers.back().request_stop();
            m_workers.pop_back();
            //--------------------------
        }// end if (workerCount > 1)
        //--------------------------
    }// end if (taskCount < m_lowerThreshold and workerCount > MIN_LIMIT)
    //--------------------------
}// end void Utils::ThreadPool::adjustWorkers(void)
//--------------------------------------------------------------
// void Utils::ThreadPool::adjustmentThreadFunction(void) {
//     //--------------------------
//     while (!m_stop.load()) {
//         //--------------------------
//         std::unique_lock<std::mutex> lock(m_mutex);
//         //--------------------------
//         m_taskAvailableCondition.wait_for(lock, CHECK_INTERVAL, [this] { return m_stop.load(); });
//         //--------------------------
//         if (m_stop.load()){
//             //--------------------------
//             return;
//             //--------------------------
//         }// end if (m_stop.load())
//         //--------------------------
//         adjustWorkers();
//         //--------------------------
//     }// end while (!m_stop.load())
//     //--------------------------
// }// end void Utils::ThreadPool::adjustmentThreadFunction(void)
//--------------------------------------------------------------
// void Utils::ThreadPool::adjustmentThreadFunction(void) {
//     //--------------------------
//     while (!m_stop.load()) {
//         //--------------------------
//         std::this_thread::sleep_for(CHECK_INTERVAL);
//         //--------------------------
//         if (m_stop.load()){
//             return;
//         }// end  if (m_stop.load())
//         //--------------------------
//         adjustWorkers();
//         //--------------------------
//     }// end while (!m_stop.load())
// }// end void Utils::ThreadPool::adjustmentThreadFunction(void)
//--------------------------------------------------------------
void Utils::ThreadPool::adjustmentThreadFunction(const std::stop_token& stoken) {
    //--------------------------
    while (!stoken.stop_requested()) {
        //--------------------------
        adjustWorkers();
        std::this_thread::sleep_for(CHECK_INTERVAL);
        //--------------------------
    }// end while (!stoken.stop_requested())
    //--------------------------
}// end void Utils::ThreadPool::adjustmentThreadFunction(const std::stop_token& stoken)
//--------------------------------------------------------------
void Utils::ThreadPool::addTask(Task&& task) {
    //--------------------------
    if (m_stop.load()) {
        throw std::runtime_error("ThreadPool is stopping, cannot add more tasks");
    }// end if (m_stop.load())
    //--------------------------
    std::scoped_lock lock(m_mutex);
    //--------------------------
    m_tasks.emplace(std::make_shared<Task>(std::move(task)));
    //--------------------------
    m_taskAvailableCondition.notify_one();
    //--------------------------
}// end void Utils::ThreadPool::addTask(Task&& task)
//--------------------------------------------------------------
size_t Utils::ThreadPool::activeWorkers(void) const{
    //--------------------------
    return m_workers.size();
    //--------------------------
}// end size_t Utils::ThreadPool::activeWorkers() const
//--------------------------------------------------------------
size_t Utils::ThreadPool::queuedTasks(void) const{
    //--------------------------
    std::unique_lock lock(m_mutex);
    //--------------------------
    return m_tasks.size();
    //--------------------------
}// end size_t Utils::ThreadPool::queuedTasks() const
//--------------------------------------------------------------
std::tuple<size_t, size_t, size_t> Utils::ThreadPool::get_status(void){
    //--------------------------
    return {m_failedTasksCount.load(), m_retriedTasksCount.load(), m_completedTasksCount.load()};
    //--------------------------
}// end std::tuple<size_t, size_t, size_t> Utils::ThreadPool::get_status(void)
//--------------------------------------------------------------
void Utils::ThreadPool::status_display_internal(void){
    //--------------------------
    auto [failedTasksCount, retriedTasksCount, completedTasksCount] = get_status();
    //--------------------------
    std::cout   << "Failed Tasks:    " << failedTasksCount  << "\n"
                << "Retried Tasks:   " << retriedTasksCount << "\n"
                << "Completed Tasks: " << completedTasksCount << std::endl;
    //--------------------------
}// end void Utils::ThreadPool::status_display_internal(void)
//--------------------------------------------------------------