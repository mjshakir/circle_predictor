#pragma once
//--------------------------------------------------------------
// cpp standard library
//--------------------------------------------------------------
#include <iostream>
#include <chrono>
#include <functional>
#include <tuple>
#include <string>
//--------------------------------------------------------------
class TimeIT{
    //--------------------------------------------------------------
    public:
        //--------------------------
        // Constructor 
        //--------------------------
        TimeIT(void);
        //--------------------------
        uint64_t get_time(void) const;
        //--------------------------
        double get_time_seconds(void) const;
        //--------------------------
        template <typename ReturnType, typename... Args>
        std::tuple<ReturnType, double> timeFunction(const std::function<ReturnType(Args...)>& function, const Args&... args){
            //--------------------------
            return time_function(function, args...);
            //--------------------------
        }// end std::tuple<ReturnType, double> timeFunction(const std::function<ReturnType(Args...)>& function, const Args&... args)
        //--------------------------
        template <typename ReturnType, typename... Args>
        ReturnType timeFunction(const std::string& function_name, const std::function<ReturnType(Args...)>& function, const Args&... args){
            //--------------------------
            return time_function(function_name, function, args...);
            //--------------------------
        }// end ReturnType timeFunction(const std::string& function_name, const std::function<ReturnType(Args...)>& function, const Args&... args)
        //--------------------------------------------------------------
    protected:
        //--------------------------
        uint64_t nanoseconds_time(void) const;
        //--------------------------
        template <typename ReturnType, typename... Args>
        std::tuple<ReturnType, double> time_function(const std::function<ReturnType(Args...)>& function, const Args&... args) {
            //--------------------------
            TimeIT timer;
            //--------------------------
            ReturnType result = function(args...);
            //--------------------------
            return {result, timer.get_time_seconds()};
            //--------------------------
        }// end std::tuple<ReturnType, double> time_function(const std::function<ReturnType(Args...)>& function, const Args&... args)
        //--------------------------
        template <typename ReturnType, typename... Args>
        ReturnType time_function(const std::string& function_name, const std::function<ReturnType(Args...)>& function, const Args&... args) {
            //--------------------------
            TimeIT timer;
            //--------------------------
            ReturnType result = function(args...);
            //--------------------------
            std::cout << "Function: " << function_name << " [" << format_duration(timer.get_time_seconds()) << "]" << std::endl;
            //--------------------------
            return result;
            //--------------------------
        }// end ReturnType time_function(const std::string& function_name, const std::function<ReturnType(const Args&...)>& function, const Args&... args)
        //--------------------------
        std::string format_duration(const double& seconds) const;
        //--------------------------------------------------------------
    private:
        //--------------------------
        std::chrono::time_point<std::chrono::high_resolution_clock> m_start;          
    //--------------------------------------------------------------

};