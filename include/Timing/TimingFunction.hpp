#pragma once
//--------------------------------------------------------------
// User defined library
//--------------------------------------------------------------
#include "Timing/Timing.hpp"
//--------------------------------------------------------------
// cpp standard library
//--------------------------------------------------------------
#include <mutex>
#include <memory>
//--------------------------------------------------------------
template<typename T>
//--------------------------------------------------------------
class TimingFunction : private Timing {
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        // Delete the constructor, destructor, copy constructor(uncloneable), and unassignable
        //--------------------------
        TimingFunction() = delete;
        //--------------------------
        TimingFunction(TimingFunction &other) = delete;
        //--------------------------
        void operator=(const TimingFunction &) = delete;
        //--------------------------------------------------------------
        TimingFunction(const T&& function) : mt_results(std::move(function)){
            //--------------------------
        }// end TimingFunction(const T&& function) : mt_results(std::move(function))
        //--------------------------
        TimingFunction(const T&& function, const std::string& function_name) : mt_results(std::move(function)){
            //--------------------------
            set_function_name(function_name);
            //--------------------------
        }// end TimingFunction(const T&& function, const std::string& function_name) : mt_results(std::move(function))
        //--------------------------
        const T& get_result(void) const {
            //--------------------------
            return mt_results;
            //--------------------------
        }// end const T& get_result(void) const
        //--------------------------------------------------------------
    private:
        //--------------------------
        T mt_results;
        //--------------------------------------------------------------
};