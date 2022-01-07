#pragma once
//--------------------------------------------------------------
// cpp standard library
//--------------------------------------------------------------
#include <iostream>
#include <chrono>
//--------------------------------------------------------------
class Timing{
    //--------------------------------------------------------------
    public:
        //--------------------------
        // Constructor 
        //--------------------------
        Timing();
        //--------------------------
        // Constructor 
        //--------------------------
        // std::string function_name: a string to pass the function name 
        //--------------------------
        // Examples of ways to pass the function name
        //------------
        // MSVC: __FUNCTION__, __FUNCDNAME__, __FUNCSIG__
        // GCC: __func__, __FUNCTION__, __PRETTY_FUNCTION__
        //--------------------------
        Timing(const std::string& function_name);
        //--------------------------
        // Destructor
        //--------------------------
        ~Timing();
        //--------------------------------------------------------------
    protected:
        //--------------------------
        void set_function_name(const std::string& function_name);
        //--------------------------
        // std::chrono::time_point_cast<std::chrono::microseconds> 
        //--------------------------------------------------------------
    private:
        //--------------------------
        std::string m_function_name;
        //--------------------------
        std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
    //--------------------------------------------------------------
};