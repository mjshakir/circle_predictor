//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Timing/TimeIT.hpp"
//--------------------------------------------------------------
// Definitions
//--------------------------------------------------------------
#define CONVERT_NANO 1e9
//--------------------------------------------------------------
TimeIT::TimeIT(void) : m_start(std::chrono::high_resolution_clock::now()) {
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
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - m_start).count();
    //--------------------------
}// end uint64_t TimeIT::nanoseconds_time(void)
//--------------------------------------------------------------