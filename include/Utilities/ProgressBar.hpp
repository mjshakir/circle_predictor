#pragma once

//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <iostream>
#include <chrono>
#include <string>
#include <mutex>
//--------------------------------------------------------------
namespace Utils{
    //--------------------------------------------------------------
    class ProgressBar {
        //--------------------------------------------------------------
        public:
            //--------------------------------------------------------------
            ProgressBar(const std::string& name = "Progress:", 
                        const std::string& progress_char = "+", 
                        const std::string& empty_space_char = "-");
            //--------------------------
            ProgressBar(const size_t& total, 
                        const std::string& name = "Progress:", 
                        const std::string& progress_char = "+", 
                        const std::string& empty_space_char = "-");
            //--------------------------
            void update(void);
            //--------------------------
            constexpr bool done(void);
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            void display(void);
            //--------------------------
            void tick(void);
            //--------------------------
            constexpr bool is_done(void);
            //--------------------------
            double get_remaining_seconds(void);
            //--------------------------
            double calculate_etc(void);
            //--------------------------
            double calculate_elapsed(void);
            //--------------------------------------------------------------
        private:
            //--------------------------------------------------------------
            size_t m_total, m_progress;
            //--------------------------
            const std::string m_name, m_progress_char, m_empty_space_char;
            //--------------------------
            bool m_is_first_tick;
            //--------------------------
            std::chrono::steady_clock::time_point m_last_tick_time, m_start_time;
            //--------------------------
            std::chrono::duration<double> m_average_delta_time;
            //--------------------------
            std::mutex m_mutex;
        //--------------------------------------------------------------
    };// end class ProgressBar
    //--------------------------------------------------------------
}//end namespace CircleEquation
//--------------------------------------------------------------