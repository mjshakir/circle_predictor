#pragma once

//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <iostream>
#include <chrono>
#include <string>
//--------------------------------------------------------------
namespace Utils{
    //--------------------------------------------------------------
    class ProgressBar {
        //--------------------------------------------------------------
        public:
            //--------------------------------------------------------------
            ProgressBar(const uint8_t& bar_length = 30U,
                        const std::string& name = "Progress:", 
                        const std::string& progress_char = "+", 
                        const std::string& empty_space_char = "-");
            //--------------------------
            ProgressBar(const size_t& total,
                        const uint8_t& bar_length = 30U,
                        const std::string& name = "Progress:", 
                        const std::string& progress_char = "+", 
                        const std::string& empty_space_char = "-");
            //--------------------------
            ProgressBar           (ProgressBar const&) = delete;
            ProgressBar& operator=(ProgressBar const&) = delete;
            ProgressBar           (ProgressBar&&)      = delete;
            ProgressBar& operator=(ProgressBar&&)      = delete;
            //--------------------------
            ~ProgressBar(void)                         = default;
            //--------------------------
            void update(void);
            //--------------------------
            constexpr bool done(void);
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            void append_elapsed_time(std::stringstream& ss);
            //--------------------------
            void append_remaining_time(std::stringstream& ss);
            //--------------------------
            void display(void);
            //--------------------------
            void tick(void);
            //--------------------------
            inline bool is_done(void) const ;
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
            const uint8_t m_bar_length;
            //--------------------------
            const std::string m_name, m_progress_char, m_empty_space_char;
            //--------------------------
            bool m_is_first_tick;
            //--------------------------
            std::chrono::steady_clock::time_point m_last_tick_time, m_start_time;
            //--------------------------
            std::chrono::duration<double> m_average_delta_time;
        //--------------------------------------------------------------
    };// end class ProgressBar
    //--------------------------------------------------------------
}//end namespace CircleEquation
//--------------------------------------------------------------