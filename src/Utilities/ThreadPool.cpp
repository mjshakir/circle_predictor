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
constexpr auto CHECK_INTERVAL = std::chrono::milliseconds(100);  // For example, check every 10 seconds
//--------------------------------------------------------------
Utils::ThreadPool::ThreadPool(const size_t& numThreads) :   m_stop(false), 
                                                            m_failedTasksCount(0),
                                                            m_retriedTasksCount(0),
                                                            m_completedTasksCount(0),
                                                            m_lowerThreshold(1UL),
                                                            m_upperThreshold(static_cast<size_t>(std::thread::hardware_concurrency())){
    //--------------------------
    auto _threads_number = std::max(std::min(m_upperThreshold, numThreads), m_lowerThreshold);
    create_task(_threads_number);
    //--------------------------
    m_tasks.reserve(numThreads);
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
        Utils::ThreadTask task;
        //--------------------------
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
            }// end if (stoken.stop_requested() and m_tasks.empty() and m_stop.load())
            //--------------------------
            if (auto opt_task = m_tasks.pop_top(); opt_task.has_value()) {
                //--------------------------
                task = std::move(opt_task.value());
                //--------------------------
            }// end if (auto opt_task = m_tasks.pop_top(); opt_task.has_value())
            //--------------------------
        }// end Append tasks
        //--------------------------
        try {
            //--------------------------
            bool _success = task.try_execute();
            //--------------------------
            if(_success){
                ++m_completedTasksCount;
            }// end if(_success)
            //--------------------------
        } // end try
        catch (const std::exception& e) {
            //--------------------------
            if (task.get_retries() > 0) {
                //--------------------------
                std::scoped_lock lock(m_mutex);
                //--------------------------
                task.decrease_retries();
                m_tasks.push(std::move(task)); // Put the modified copy of the task back in the queue
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
            if (task.get_retries() > 0) {
                //--------------------------
                std::scoped_lock lock(m_mutex);
                //--------------------------
                task.decrease_retries();
                m_tasks.push(std::move(task)); // Put the modified copy of the task back in the queue
                //--------------------------
                ++m_retriedTasksCount;
                //--------------------------
            } else {
                //--------------------------
                ++m_failedTasksCount;
                //--------------------------
                std::cerr << "Unknown error occurred in task." << std::endl;
                //--------------------------
            }// end else
            //--------------------------
        }// end catch (...)
        //--------------------------
    }// end while (!stoken.stop_requested())
    //--------------------------
}// end void Utils::ThreadPool::workerFunction(void)
//--------------------------------------------------------------
void Utils::ThreadPool::stop(void){
    //--------------------------
    //  { // set m_stop to true
    //     //--------------------------
    //     std::scoped_lock lock(m_mutex);
    //     m_stop.store(true);
    //     //--------------------------
    // }// end set
    // //--------------------------
    // m_taskAvailableCondition.notify_all();
    //--------------------------
    m_stop.store(true);
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
// void Utils::ThreadPool::adjustWorkers(void) {
//     //--------------------------
//     m_tasks.remove();
//     //--------------------------
//     size_t taskCount, workerCount;
//     //--------------------------
//     {
//         //--------------------------
//         std::scoped_lock lock(m_mutex);
//         taskCount = queuedTasks();
//         workerCount = activeWorkers();
//         //--------------------------
//     }
//     //--------------------------
//     if (taskCount > m_upperThreshold and workerCount < m_upperThreshold) {
//         //--------------------------
//         auto threadsToCreate = std::min(std::max(static_cast<size_t>(0), taskCount - m_upperThreshold),
//                                         m_upperThreshold - workerCount);
//         //--------------------------
//         threadsToCreate = std::min(threadsToCreate,
//                                    std::max(static_cast<size_t>(0), m_upperThreshold - workerCount));
//         //--------------------------
//         create_task((threadsToCreate > 0) ? threadsToCreate : m_lowerThreshold);
//         //--------------------------
//     }// end if (taskCount > m_upperThreshold and workerCount < MAX_LIMIT)
//     //--------------------------
//     if (taskCount < m_lowerThreshold and workerCount > m_lowerThreshold) {
//         //--------------------------
//         if (workerCount > 1) {
//             //--------------------------
//             m_workers.back().request_stop();
//             m_workers.pop_back();
//             //--------------------------
//         }// end if (workerCount > 1)
//         //--------------------------
//     }// end if (taskCount < m_lowerThreshold and workerCount > MIN_LIMIT)
//     //--------------------------
// }// end void Utils::ThreadPool::adjustWorkers(void)
//--------------------------------------------------------------
void Utils::ThreadPool::adjustWorkers(void) {
    //--------------------------
    m_tasks.remove();
    //--------------------------
    size_t taskCount, workerCount;
    //--------------------------
    {
        //--------------------------
        std::scoped_lock lock(m_mutex);
        taskCount = queuedTasks();
        workerCount = activeWorkers();
        //--------------------------
    }
    //--------------------------
    if (taskCount < m_lowerThreshold and workerCount > m_lowerThreshold) {
        //--------------------------
        if (workerCount > 1) {
            //--------------------------
            m_workers.back().request_stop();
            m_workers.pop_back();
            //--------------------------
        }// end if (workerCount > 1)
        //--------------------------
        if (taskCount > 0U) {
        //--------------------------
            m_workers.clear();
        //--------------------------
        }// end if (taskCount > 0U)
    //--------------------------
    }// end if (taskCount < m_lowerThreshold and workerCount > m_lowerThreshold)
    //--------------------------
    if (taskCount < m_upperThreshold and taskCount > m_lowerThreshold) {
        //--------------------------
        create_task(taskCount);
        //--------------------------
    }// end if (taskCount < m_upperThreshold and taskCount > m_lowerThreshold)
    //--------------------------
    if (taskCount > m_upperThreshold) {
        //--------------------------
        create_task(m_upperThreshold);
        //--------------------------
    }// end if (taskCount > m_upperThreshold)
    //--------------------------
}// end void Utils::ThreadPool::adjustWorkers(void)
//--------------------------------------------------------------
// void Utils::ThreadPool::adjustWorkers(void) {
//     // Remove finished tasks
//     m_tasks.remove();
    
//     size_t taskCount, workerCount;
//     {
//         std::scoped_lock lock(m_mutex);
//         taskCount = queuedTasks();
//         workerCount = activeWorkers();
//     }

//     // Calculate the difference between current workers and desired number of workers
//     size_t desiredWorkerCount = (taskCount > m_upperThreshold) ? m_upperThreshold 
//                           : (taskCount < m_lowerThreshold) ? m_lowerThreshold 
//                           : workerCount; 

//     if (workerCount < desiredWorkerCount) {
//         size_t threadsToCreate = desiredWorkerCount - workerCount;
//         create_task(threadsToCreate);
//     }
//     else if (workerCount > desiredWorkerCount) {
//         size_t threadsToRemove = workerCount - desiredWorkerCount;
//         for (size_t i = 0; i < threadsToRemove; ++i) {
//             if (!m_workers.empty()) {
//                 m_workers.back().request_stop();
//                 m_workers.pop_back();
//             }
//         }
//     }
// }
//--------------------------------------------------------------
void Utils::ThreadPool::adjustmentThreadFunction(const std::stop_token& stoken) {
    //--------------------------
    while (!stoken.stop_requested()) {
        //--------------------------
        std::this_thread::sleep_for(CHECK_INTERVAL);
        adjustWorkers();
        //--------------------------
    }// end while (!stoken.stop_requested())
    //--------------------------
}// end void Utils::ThreadPool::adjustmentThreadFunction(const std::stop_token& stoken)
//--------------------------------------------------------------
void Utils::ThreadPool::addTask(Utils::ThreadTask&& task) {
    //--------------------------
    if (m_stop.load()) {
        throw std::runtime_error("ThreadPool is stopping, cannot add more tasks");
    }// end if (m_stop.load())
    //--------------------------
    std::scoped_lock lock(m_mutex);
    //--------------------------
    m_tasks.push(std::move(task));
    //--------------------------
    m_taskAvailableCondition.notify_one();
    //--------------------------
}// end void Utils::ThreadPool::addTask(Task&& task)
//--------------------------------------------------------------
size_t Utils::ThreadPool::activeWorkers(void) const{
    //--------------------------
    return m_workers.size();
    //--------------------------
}// end constexpr size_t Utils::ThreadPool::activeWorkers(void) const
//--------------------------------------------------------------
size_t Utils::ThreadPool::queuedTasks(void) const{
    //--------------------------
    std::unique_lock lock(m_mutex);
    //--------------------------
    return m_tasks.size();
    //--------------------------
}// end constexpr size_t Utils::ThreadPool::queuedTasks(void) const
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