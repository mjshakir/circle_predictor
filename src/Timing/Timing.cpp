//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Timing/Timing.hpp"
//--------------------------------------------------------------
// Definitions
//--------------------------------------------------------------
// ANSI colors
//--------------------------
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
//--------------------------------------------------------------
Timing::Timing() : m_function_name("Unknown"), m_start(std::chrono::high_resolution_clock::now()) {
    //--------------------------
}// end Timing::Timing()
//--------------------------------------------------------------
Timing::Timing(const std::string& function_name) : m_function_name(function_name), m_start(std::chrono::high_resolution_clock::now()) {
    //--------------------------
}// end Timing::Timing(const std::string& function_name)
//--------------------------------------------------------------
Timing::~Timing(){
    //--------------------------
    auto end = std::chrono::high_resolution_clock::now();
    //--------------------------
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - m_start);
    //--------------------------
    printf(ANSI_COLOR_GREEN "%s" ANSI_COLOR_RESET " Execution time: [" ANSI_COLOR_RED "%f " ANSI_COLOR_RESET "S]\n", 
            m_function_name.c_str(), duration.count()/CONVERT_NANO);
    //--------------------------
}// end Timing::~Timing()
//--------------------------------------------------------------
void Timing::set_function_name(const std::string& function_name){
    //--------------------------
    m_function_name = function_name;
    //--------------------------
}// end void Timing::set_function_name(const std::string& function_name)
//--------------------------------------------------------------