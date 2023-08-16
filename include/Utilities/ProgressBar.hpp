#pragma once

//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <iostream>
#include <chrono>
#include <string>
//--------------------------------------------------------------
// Boost library 
//--------------------------------------------------------------
#include <boost/circular_buffer.hpp>
//--------------------------------------------------------------
namespace Utils{
    //--------------------------------------------------------------
    class ProgressBar {
        //--------------------------------------------------------------
        public:
            //--------------------------------------------------------------
            ProgressBar(const std::string& name = "Progress:", 
                        const std::string& progress_char = "#", 
                        const std::string& empty_space_char = "-");
            //--------------------------
            ProgressBar(const size_t& total,
                        const std::string& name = "Progress:", 
                        const std::string& progress_char = "#", 
                        const std::string& empty_space_char = "-");
            //--------------------------
            ProgressBar           (ProgressBar const&) = delete;
            ProgressBar& operator=(ProgressBar const&) = delete;
            //--------------------------
            ProgressBar           (ProgressBar&&)      = delete;
            ProgressBar& operator=(ProgressBar&&)      = delete;
            //--------------------------
            ~ProgressBar(void)                         = default;
            //--------------------------
            void update(void);
            //--------------------------
            bool done(void);
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            #ifndef HAVE_FMT
                //--------------------------
                void append_time(std::stringstream& ss, double time, const std::string& label);
                //--------------------------
            #endif
            //--------------------------
            #ifdef HAVE_FMT
                //--------------------------
                std::string append_time(double time, const std::string& label);
                //--------------------------
            #endif
            //--------------------------
            void display(void);
            //--------------------------
            void tick(void);
            //--------------------------
            bool is_done(void) const ;
            //--------------------------
            double calculate_etc(void);
            //--------------------------
            double calculate_elapsed(void);
            //--------------------------
            static size_t get_terminal_width(void);
            //--------------------------
            static void calculate_bar(void); 
            //--------------------------------------------------------------
        private:
            //--------------------------------------------------------------
            size_t m_total, m_progress, m_update_counter;
            //--------------------------
            const std::string m_name, m_progress_char, m_empty_space_char;
            //--------------------------
            boost::circular_buffer<double> m_delta_times;
            //--------------------------
            double m_last_etc;
            //--------------------------
            std::chrono::steady_clock::time_point m_last_tick_time, m_start_time;
            //--------------------------
            static size_t  m_name_length, m_bar_length, m_available_width, m_spaces_after_bar;
            //--------------------------
            static void handle_winch_signal(int signum);
        //--------------------------------------------------------------
    };// end class ProgressBar
    //--------------------------------------------------------------
}//end namespace CircleEquation
//--------------------------------------------------------------