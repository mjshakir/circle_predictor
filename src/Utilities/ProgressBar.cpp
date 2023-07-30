//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Utilities/ProgressBar.hpp"
//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <sstream>
#include <iomanip>
#include <climits>
//--------------------------------------------------------------
// Definitions 
//--------------------------------------------------------------
#define ALPHA 0.1
//--------------------------
// ANSI Colors
//--------------------------
#define ANSI_COLOR_BLACK   "\x1b[30m"
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_WHITE   "\x1b[37m"
#define ANSI_COLOR_RESET   "\x1b[0m"
//--------------------------
// ANSI Bold
//--------------------------
#define ANSI_BOLD_ON       "\x1b[1m"
#define ANSI_BOLD_OFF      ANSI_COLOR_RESET
//--------------------------------------------------------------
Utils::ProgressBar::ProgressBar(const std::string& name, 
                                const std::string& progress_char, 
                                const std::string& empty_space_char) :  m_total(std::numeric_limits<size_t>::max()), 
                                                                        m_progress(0), 
                                                                        m_name(name), 
                                                                        m_progress_char(progress_char), 
                                                                        m_empty_space_char(empty_space_char), 
                                                                        m_is_first_tick(true){
    //--------------------------
}//end Utils::ProgressBar::ProgressBar
//--------------------------------------------------------------
Utils::ProgressBar::ProgressBar(const size_t& total, 
                                const std::string& name, 
                                const std::string& progress_char, 
                                const std::string& empty_space_char) :  m_total(total), 
                                                                        m_progress(0), 
                                                                        m_name(name), 
                                                                        m_progress_char(progress_char), 
                                                                        m_empty_space_char(empty_space_char), 
                                                                        m_is_first_tick(true){
    //--------------------------
}// end Utils::ProgressBar::ProgressBar
//--------------------------------------------------------------
void Utils::ProgressBar::update(void){
    //--------------------------
    tick();
    //--------------------------
}//end void Utils::ProgressBar::update(void)
//--------------------------------------------------------------
constexpr bool Utils::ProgressBar::done(void){
    //--------------------------
    return is_done();
    //--------------------------
}// end constexpr bool Utils::ProgressBar::done(void)
//--------------------------------------------------------------
double Utils::ProgressBar::get_remaining_seconds(void){
    //--------------------------
    std::chrono::duration<double> remaining_seconds = m_average_delta_time * (m_total - m_progress);
    return remaining_seconds.count();
    //--------------------------
}// end constexpr bool Utils::ProgressBar::done(void)
//--------------------------------------------------------------
double Utils::ProgressBar::calculate_etc(void) {
    //--------------------------
    auto current_time = std::chrono::steady_clock::now();
    auto current_delta_time = std::chrono::duration<double>(current_time - m_last_tick_time).count();
    //--------------------------
    if (m_progress > 0 and m_total > 0) {
        //--------------------------
        // calculate EMA of time per unit of work
        double average_delta_time_seconds = m_average_delta_time.count();
        average_delta_time_seconds = ALPHA * current_delta_time + (1 - ALPHA) * average_delta_time_seconds;
        m_average_delta_time = std::chrono::duration<double>(average_delta_time_seconds);
        //--------------------------
        // ETC = average time per unit of work * remaining work
        return m_average_delta_time.count() * (m_total - m_progress);
        //--------------------------
    }//end if (m_progress > 0 and m_total > 0)
    //--------------------------
    // ETC is indeterminable when no progress has been made.
    return std::numeric_limits<double>::max();
    //--------------------------
}// end double Utils::ProgressBar::calculate_etc(void) 
//--------------------------------------------------------------
double Utils::ProgressBar::calculate_elapsed(void) {
    //--------------------------
    auto now = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_start_time).count();
    //--------------------------
    return elapsed_time / 1000.0;  // Convert milliseconds to seconds
    //--------------------------
}// end double Utils::ProgressBar::calculate_elapsed(void)
//--------------------------------------------------------------
void Utils::ProgressBar::display(void) {
    //--------------------------
    std::stringstream ss;
    //--------------------------
    constexpr uint8_t bar_length = 30;
    //--------------------------
    std::string bar(bar_length, m_empty_space_char[0]);
    //--------------------------
    size_t position = 0, percent = 0;
    //--------------------------
    double ratio;
    //--------------------------
    // Carriage return at the start, not after the name
    ss << '\r';
    //--------------------------
    if(m_total != 0 and m_total != std::numeric_limits<size_t>::max()) {
        //--------------------------
        ratio = static_cast<double>(m_progress) / m_total;
        percent = static_cast<size_t>(ratio * 100);
        position = bar_length * ratio;
        //--------------------------
        ss << m_name << ": " << std::setw(3) << percent << "% [" << ANSI_BOLD_ON;  // Start bold
        //--------------------------
        for (size_t i = 0; i < position; i++) {
            bar[i] = m_progress_char[0];
        }
        //--------------------------
        // Color the completed part green, the rest white
        ss << ANSI_COLOR_GREEN << bar.substr(0, position) << ANSI_COLOR_RESET;  // End green color
        ss << ANSI_COLOR_RED << bar.substr(position) << ANSI_COLOR_RESET;  // End white color
        //--------------------------
    } // end if(m_total != 0 && m_total != std::numeric_limits<size_t>::max())
    else {
        //--------------------------
        // for unknown total
        //--------------------------
        position = m_progress % bar_length;
        bar[position] = m_progress_char[0];
        ss << m_name << ": " << "[" << ANSI_BOLD_ON;  // Start bold
        //--------------------------
        // Color the completed part green, the rest white
        ss << ANSI_COLOR_GREEN << bar.substr(0, position) << ANSI_COLOR_RESET;  // End green color
        ss << ANSI_COLOR_RED << bar.substr(position) << ANSI_COLOR_RESET;  // End white color
        //--------------------------
    }// end else
    //--------------------------
    ss << "] " << ANSI_BOLD_OFF;  // End bold;
    //--------------------------
    // Calculate elapsed time
    //--------------------------
    auto elapsed_time = calculate_elapsed();
    auto elapsed_hours = static_cast<size_t>(elapsed_time / 3600);
    elapsed_time -= elapsed_hours * 3600;
    auto elapsed_minutes = static_cast<size_t>(elapsed_time / 60);
    elapsed_time -= elapsed_minutes * 60;
    auto elapsed_seconds = static_cast<size_t>(elapsed_time);
    elapsed_time -= elapsed_seconds;
    auto elapsed_milliseconds = static_cast<size_t>(elapsed_time * 1000);
    //--------------------------
    // Append elapsed time to the stringstream
    //--------------------------
    ss << "Elapsed: " << std::setw(2) << std::setfill('0') << elapsed_hours << ":"
       << std::setw(2) << std::setfill('0') << elapsed_minutes << ":"
       << std::setw(2) << std::setfill('0') << elapsed_seconds << ":"
       << std::setw(3) << std::setfill('0') << elapsed_milliseconds << " ";
    //--------------------------
    // Only calculate ETC if total is known
    if(m_total != 0 and m_total != std::numeric_limits<size_t>::max()) {
        auto remaining_seconds_count = calculate_etc();
        auto hours = static_cast<size_t>(remaining_seconds_count) / 3600;
        auto minutes = static_cast<size_t>((remaining_seconds_count - hours * 3600) / 60);
        auto seconds = static_cast<size_t>(remaining_seconds_count) % 60;
        auto milliseconds = static_cast<size_t>((remaining_seconds_count - static_cast<size_t>(remaining_seconds_count)) * 1000);
        auto days = hours / 24;
        hours %= 24;
        ss << "ETC: " << days << ":" << std::setw(2) << std::setfill('0') << hours << ":" 
        << std::setw(2) << std::setfill('0') << minutes << ":" 
        << std::setw(2) << std::setfill('0') << seconds << ":" 
        << std::setw(3) << std::setfill('0') << milliseconds << " ";
    }// end if(m_total != 0 && m_total != std::numeric_limits<size_t>::max())
    //--------------------------
    std::cout << ss.str();
    std::cout.flush();
    //--------------------------
}// end void ProgressBar::display(void) 
//--------------------------------------------------------------
void Utils::ProgressBar::tick(void) {
    //--------------------------
    std::lock_guard<std::mutex> lock(m_mutex);
    //--------------------------
    if (!m_is_first_tick) {
        //--------------------------
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> delta_time = now - m_last_tick_time;
        m_average_delta_time = ALPHA * delta_time + (1 - ALPHA) * m_average_delta_time;
        m_last_tick_time = now;
        //--------------------------
    } // end if (!m_is_first_tick)
    else {
        //--------------------------
        m_start_time = std::chrono::steady_clock::now();
        m_last_tick_time = m_start_time;
        m_is_first_tick = false;
        //--------------------------
    }// end else
    //--------------------------
    // Check if progress is done and total is known, if so, prevent progress increment
    if(!(m_total != std::numeric_limits<size_t>::max() && is_done())) {
        //--------------------------
        ++m_progress;
        //--------------------------
    }// end if(!(m_total != std::numeric_limits<size_t>::max() && is_done()))
    //--------------------------
    display();
    //--------------------------
}// end void Utils::ProgressBar::tick(void)
//--------------------------------------------------------------
// void Utils::ProgressBar::tick(void) {
//     //--------------------------
//     std::lock_guard<std::mutex> lock(m_mutex);
//     //--------------------------
//     if (!m_is_first_tick) {
//         //--------------------------
//         auto now = std::chrono::steady_clock::now();
//         auto current_delta_time = std::chrono::duration<double>(now - m_last_tick_time);
//         // Convert durations to double
//         double current_delta_time_seconds = current_delta_time.count();
//         double average_delta_time_seconds = m_average_delta_time.count();

//         // Calculate EMA of time per unit of work
//         average_delta_time_seconds = ALPHA * current_delta_time_seconds + (1 - ALPHA) * average_delta_time_seconds;
//         m_average_delta_time = std::chrono::duration<double>(average_delta_time_seconds);

//         m_last_tick_time = now;
//         //--------------------------
//     } else {
//         //--------------------------
//         m_start_time = std::chrono::steady_clock::now();
//         m_last_tick_time = m_start_time;
//         m_is_first_tick = false;
//         //--------------------------
//     }// end else
//     //--------------------------
//     // Check if progress is done and total is known, if so, prevent progress increment
//     if(!(m_total != std::numeric_limits<size_t>::max() && is_done())) {
//         //--------------------------
//         ++m_progress;
//         //--------------------------
//     }// end if(!(m_total != std::numeric_limits<size_t>::max() && is_done()))
//     //--------------------------
//     display();
//     //--------------------------
// }// end void Utils::ProgressBar::tick(void)
//--------------------------------------------------------------
constexpr bool Utils::ProgressBar::is_done(void){
    //--------------------------
    return m_progress >= m_total;
    //--------------------------
}// end bool Utils::ProgressBar::is_done(void)
//--------------------------------------------------------------