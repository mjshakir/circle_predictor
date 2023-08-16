//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Timing/TimeIT.hpp"
//--------------------------------------------------------------
// Definitions
//--------------------------------------------------------------
constexpr double CONVERT_NANO   = 1e9;
//--------------------------
constexpr double NANOSECOND     = 1e-9;
constexpr double MICROSECOND    = 1e-6;
constexpr double MILLISECOND    = 1e-3;
constexpr double SECOND         = 1.0;
constexpr double MINUTE         = 60.0;
constexpr double HOUR           = 3600.0;
constexpr double DAY            = 86400.0;
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
std::string TimeIT::format_duration(const double& seconds) const{
    //--------------------------
    if (seconds < MICROSECOND) {
        //--------------------------
        return std::to_string((seconds / (1/NANOSECOND))) + " nanoseconds";
        //--------------------------
    } // if (seconds < MICROSECOND)
    //--------------------------
    if (seconds < MILLISECOND) {
        //--------------------------
        return std::to_string((seconds / (1/MICROSECOND))) + " microseconds";
        //--------------------------
    } // end if (seconds < MILLISECOND)
    //--------------------------
    if (seconds < SECOND) {
        //--------------------------
        return std::to_string((seconds / (1/MILLISECOND))) + " milliseconds";
        //--------------------------
    } // end if (seconds < SECOND)
    //--------------------------
    if (seconds < MINUTE) {
        //--------------------------
        return std::to_string(seconds) + " seconds";
        //--------------------------
    } // end if (seconds < MINUTE)
    //--------------------------
    if (seconds < HOUR) {
        //--------------------------
        return std::to_string((seconds / MINUTE)) + " minutes";
        //--------------------------
    } // end if (seconds < HOUR)
    //--------------------------
    if (seconds < DAY) {
        //--------------------------
        return std::to_string((seconds / HOUR)) + " hours";
        //--------------------------
    } // end if (seconds < DAY)
    //--------------------------
    return std::to_string((seconds / DAY)) + " days";
    //--------------------------
}// end std::string TimeIT::format_duration(const double& seconds) const
//--------------------------------------------------------------