//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Utilities/ThreadTask.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <limits>
//--------------------------------------------------------------
// public
//--------------------------
void Utils::ThreadTask::execute(void){
    //--------------------------
    execute_local();
    //--------------------------
}// end void Utils::ThreadTask::execute(void)
//--------------------------------------------------------------
bool Utils::ThreadTask::try_execute(void){
    //--------------------------
    return try_execute_local();
    //--------------------------
}// end bool Utils::ThreadTask::try_execute(void)
//--------------------------------------------------------------
std::future<std::any> Utils::ThreadTask::get_future(void) {
    //--------------------------
    return get_future_local();
    //--------------------------
}// end std::any Utils::ThreadTask::get_result(void) const
//--------------------------------------------------------------
bool Utils::ThreadTask::is_done(void) const {
    //--------------------------
    return is_done_local();
    //--------------------------
}// end bool Utils::ThreadTask::is_done(void) const
//--------------------------------------------------------------
uint8_t Utils::ThreadTask::get_retries(void) const {
    //--------------------------
    return get_retries_local();
    //--------------------------
}// end uint8_t Utils::ThreadTask::get_retries(void) const
//--------------------------
uint8_t Utils::ThreadTask::get_priority(void) const {
    //--------------------------
    return get_priority_local();
    //--------------------------
}// end uint8_t Utils::ThreadTask::get_priority(void) const
//--------------------------------------------------------------
void Utils::ThreadTask::increase_retries(const uint8_t& amount){
    //--------------------------
    increase_retries_local(amount);
    //--------------------------
}// end void Utils::ThreadTask::increase_retries(const uint8_t& amount)
//--------------------------------------------------------------
void Utils::ThreadTask::decrease_retries(const uint8_t& amount){
    //--------------------------
    decrease_retries_local(amount);
    //--------------------------
}// end void Utils::ThreadTask::decrease_retries(const uint8_t& amount)
//--------------------------------------------------------------
void Utils::ThreadTask::increase_retries(void) {
    //--------------------------
    increase_retries_local(1);
    //--------------------------
}// end void Utils::ThreadTask::increase_retries(void)
//--------------------------------------------------------------
void Utils::ThreadTask::decrease_retries(void) {
    //--------------------------
    decrease_retries_local(1);
    //--------------------------
}// end void Utils::ThreadTask::decrease_retries(void)
//--------------------------------------------------------------
void Utils::ThreadTask::increase_priority(const uint8_t& amount){
    //--------------------------
    increase_priority_local(amount);
    //--------------------------
}// end void Utils::ThreadTask::increase_priority(const uint8_t& amount)
//--------------------------------------------------------------
void Utils::ThreadTask::decrease_priority(const uint8_t& amount){
    //--------------------------
    decrease_priority_local(amount);
    //--------------------------
}// end void Utils::ThreadTask::decrease_priority(const uint8_t& amount)
//--------------------------------------------------------------
void Utils::ThreadTask::increase_priority(void) {
    increase_priority_local(1);
}
//--------------------------------------------------------------
void Utils::ThreadTask::decrease_priority(void) {
    decrease_priority_local(1);
}
//--------------------------------------------------------------
uint8_t Utils::ThreadTask::get_status(void) const {
    return static_cast<uint8_t>(m_state);
}// end uint8_t Utils::ThreadTask::get_status(void) const
//--------------------------------------------------------------
// protected
//--------------------------
void Utils::ThreadTask::execute_local(void){
    //--------------------------
    if (m_retries == 0) { 
        return;
    }// endif (m_retries == 0)
    //--------------------------
    do {
        //--------------------------
        if (try_execute()) {
            break;
        }// end if (try_execute())
        //--------------------------
    } while (m_retries > 0);
    //--------------------------
}// end void Utils::ThreadTask::execute_local(void)
//--------------------------------------------------------------
bool Utils::ThreadTask::try_execute_local(void){
    //--------------------------
    std::any result;
    //--------------------------
    try {
        result = m_function();
    } // end try
    catch (...) {
        //--------------------------
        decrease_retries();
        return false;
        //--------------------------
    }// end catch (...)
    //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    //--------------------------
    m_promise.set_value(result);
    m_state = Utils::ThreadTask::TaskState::COMPLETED;
    //--------------------------
    return true;
    //--------------------------
}// end bool Utils::ThreadTask::try_execute_local(void)
//--------------------------------------------------------------
std::future<std::any> Utils::ThreadTask::get_future_local(void){
    //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    //--------------------------
    if (m_state ==  Utils::ThreadTask::TaskState::RETRIEVED) {
        throw std::logic_error("Future already retrieved!");
    }// end if (m_state ==  Utils::ThreadTask::TaskState::Retrieved)
    //--------------------------
    if (m_state ==  Utils::ThreadTask::TaskState::PENDING) {
        throw std::logic_error("Task not yet executed!");
    }// end if (m_state ==  Utils::ThreadTask::TaskState::Retrieved)
    //--------------------------
    m_state = Utils::ThreadTask::TaskState::RETRIEVED;
    //--------------------------
    return m_promise.get_future();
    //--------------------------
}// end std::any Utils::ThreadTask::get_result_local(void) const
//--------------------------------------------------------------
bool Utils::ThreadTask::is_done_local(void) const{
    //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    //--------------------------
    if (Utils::ThreadTask::is_void_function<decltype(m_function)>() and (m_state == Utils::ThreadTask::TaskState::COMPLETED)) {
        //--------------------------
        return true;
        //--------------------------
    }// end if (m_state == Utils::ThreadTask::TaskState::COMPLETED)
    //--------------------------
    return m_state == Utils::ThreadTask::TaskState::RETRIEVED;
    //--------------------------
}// end bool Utils::ThreadTask::is_done_local(void) const
//--------------------------------------------------------------
uint8_t Utils::ThreadTask::get_retries_local(void) const {
     //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_retries;
    //--------------------------
}// end uint8_t Utils::ThreadTask::get_retries_local(void) const
//--------------------------------------------------------------
uint8_t Utils::ThreadTask::get_priority_local(void) const {
     //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_priority;
    //--------------------------
}// end uint8_t Utils::ThreadTask::get_priority_local(void) const
//--------------------------------------------------------------
void Utils::ThreadTask::increase_retries_local(const uint8_t& amount) {
    //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    //--------------------------
    auto _results = m_retries + amount;
    //--------------------------
    m_retries = (_results > std::numeric_limits<uint8_t>::max()) ? std::numeric_limits<uint8_t>::max() : _results;
    //--------------------------
}// end void Utils::ThreadTask::increase_retries_local(const uint8_t& amount) const
//--------------------------------------------------------------
void Utils::ThreadTask::decrease_retries_local(const uint8_t& amount) {
    //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    //--------------------------
    auto _results = m_retries - amount;
    //--------------------------
    m_retries = (_results < 0U) ? 0U : _results;
    //--------------------------
}// end void Utils::ThreadTask::decrease_retries_local(const uint8_t& amount) const
//--------------------------------------------------------------
void Utils::ThreadTask::increase_priority_local(const uint8_t& amount) {
    //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    //--------------------------
    auto _results = m_priority + amount;
    //--------------------------
    m_priority = (_results > std::numeric_limits<uint8_t>::max()) ? std::numeric_limits<uint8_t>::max() : _results;
    //--------------------------
}// end void Utils::ThreadTask::increase_priority_local(const uint8_t& amount) const
//--------------------------------------------------------------
void Utils::ThreadTask::decrease_priority_local(const uint8_t& amount) {
    //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    //--------------------------
    auto _results = m_priority - amount;
    //--------------------------
    m_priority = (_results < 0U) ? 0U : _results;
    //--------------------------
}// end void Utils::ThreadTask::decrease_priority_local(const uint8_t& amount) const
//--------------------------------------------------------------