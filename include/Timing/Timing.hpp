#pragma once
//--------------------------------------------------------------
// cpp standard library
//--------------------------------------------------------------
#include <iostream>
#include <chrono>
#include <functional>
//--------------------------------------------------------------
class Timing{
    //--------------------------------------------------------------
    public:
        //--------------------------
        // Constructor 
        //--------------------------
        Timing(void);
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
        //--------------------------
        template <typename ReturnType, typename... Args>
        ReturnType timeFunction(const std::function<ReturnType(Args...)>& function, const Args&... args){
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
        void set_function_name(const std::string& function_name);
        //--------------------------
        template <typename ReturnType, typename... Args>
        ReturnType time_function(const std::function<ReturnType(Args...)>& function, const Args&... args) {
            //--------------------------
            Timing timer;
            //--------------------------
            return function(args...);
            //--------------------------
        }// end ReturnType time_function(const std::function<ReturnType(Args...)>& function, const Args&... args)
        //--------------------------
        template <typename ReturnType, typename... Args>
        ReturnType time_function(const std::string& function_name, const std::function<ReturnType(Args...)>& function, const Args&... args) {
            //--------------------------
            Timing timer(function_name);
            //--------------------------
            return function(args...);
            //--------------------------
        }// end ReturnType time_function(const std::string& function_name, const std::function<ReturnType(const Args&...)>& function, const Args&... args)
        //--------------------------------------------------------------
    private:
        //--------------------------
        std::string m_function_name;
        //--------------------------
        const std::chrono::time_point<std::chrono::high_resolution_clock> m_start;  
    //--------------------------------------------------------------
};