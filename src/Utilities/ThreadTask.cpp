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
    try_execute_local();
    //--------------------------
}// end bool Utils::ThreadTask::try_execute(void)
//--------------------------------------------------------------
std::any Utils::ThreadTask::get_result(void) const {
    //--------------------------
    return get_result_local();
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
// protected
//--------------------------
void Utils::ThreadTask::execute_local(void){
    //--------------------------
    if (m_retries == 0) { 
        return;
    }// endif (m_retries == 0)
    //--------------------------
    while (m_retries > 0) {
        //--------------------------
        if (try_execute()) {
            break;
        }// end if (try_execute())
        //--------------------------
    }// end while (m_retries > 0)
    //--------------------------
}// end void Utils::ThreadTask::execute_local(void)
//--------------------------------------------------------------
bool Utils::ThreadTask::try_execute_local(void){
    //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    //--------------------------
    try {
        //--------------------------
        m_result = m_function();
        m_done = true;
        m_condition.notify_all();
        return m_done;
        //--------------------------
    } // end try
    catch (...) {
        //--------------------------
        decrease_retries();
        return false;
        //--------------------------
    }// end catch (...)
    //--------------------------
}// end bool Utils::ThreadTask::try_execute_local(void)
//--------------------------------------------------------------
std::any Utils::ThreadTask::get_result_local(void) const{
    //--------------------------
    std::unique_lock lock(m_mutex);
    //--------------------------
    m_condition.wait(lock, [this]{ return m_done; });
    //--------------------------
    return m_result;
    //--------------------------
}// end std::any Utils::ThreadTask::get_result_local(void) const
//--------------------------------------------------------------
bool Utils::ThreadTask::is_done_local(void) const{
    //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_done;
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
    m_retries = std::min(static_cast<int16_t>(std::numeric_limits<uint8_t>::max()), static_cast<int16_t>(m_retries + amount));
    //--------------------------
}// end void Utils::ThreadTask::increase_retries_local(const uint8_t& amount) const
//--------------------------------------------------------------
void Utils::ThreadTask::decrease_retries_local(const uint8_t& amount) {
    //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    //--------------------------
    if(m_retries > 0){
        m_retries = std::max(static_cast<int16_t>(std::numeric_limits<uint8_t>::min()), static_cast<int16_t>(m_retries - amount));
    }// end if(m_retries > 0)
    //--------------------------
}// end void Utils::ThreadTask::decrease_retries_local(const uint8_t& amount) const
//--------------------------------------------------------------
void Utils::ThreadTask::increase_priority_local(const uint8_t& amount) {
    //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    m_priority = std::min(static_cast<int16_t>(std::numeric_limits<uint8_t>::max()), static_cast<int16_t>(m_priority + amount));
    //--------------------------
}// end void Utils::ThreadTask::increase_priority_local(const uint8_t& amount) const
//--------------------------------------------------------------
void Utils::ThreadTask::decrease_priority_local(const uint8_t& amount) {
    //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    //--------------------------
    if(m_priority > 0){
        m_priority = std::max(static_cast<int16_t>(std::numeric_limits<uint8_t>::min()), static_cast<int16_t>(m_priority - amount));
    }// end if(m_priority > 0)
    //--------------------------
}// end void Utils::ThreadTask::decrease_priority_local(const uint8_t& amount) const
//--------------------------------------------------------------