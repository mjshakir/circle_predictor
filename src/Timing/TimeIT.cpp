//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Timing/TimeIT.hpp"
//--------------------------------------------------------------

TimeIT::TimeIT() : m_start(std::chrono::high_resolution_clock::now()) {
    //--------------------------
}// end Timing::Timing()
//--------------------------------------------------------------
uint64_t TimeIT::get_time(void){
    //--------------------------
    return nanoseconds_time();
    //--------------------------
}// end uint64_t TimeIT::nanoseconds_time(void)
//--------------------------------------------------------------
uint64_t TimeIT::nanoseconds_time(void){
    //--------------------------
    auto end = std::chrono::high_resolution_clock::now();
    //--------------------------
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - m_start);
    //--------------------------
    return duration.count();
    //--------------------------
}// end uint64_t TimeIT::nanoseconds_time(void)
//--------------------------------------------------------------