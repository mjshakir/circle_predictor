//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Timing/TimeIT.hpp"
//--------------------------------------------------------------

TimeIT::TimeIT() : m_start(std::chrono::high_resolution_clock::now()) {
    //--------------------------
}// end Timing::Timing()
//--------------------------------------------------------------
uint64_t TimeIT::get_time(void) const{
    //--------------------------
    return nanoseconds_time();
    //--------------------------
}// end uint64_t TimeIT::get_time(void)
//--------------------------------------------------------------
double TimeIT::get_time_seconds(void) const{
    //--------------------------
    return static_cast<double>(nanoseconds_time())/CONVERT_NANO;
    //--------------------------
}// end double TimeIT::get_time_seconds(void)
//--------------------------------------------------------------
uint64_t TimeIT::nanoseconds_time(void) const{
    //--------------------------
    auto end = std::chrono::high_resolution_clock::now();
    //--------------------------
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - m_start);
    //--------------------------
    return duration.count();
    //--------------------------
}// end uint64_t TimeIT::nanoseconds_time(void)
//--------------------------------------------------------------