//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Utilities/ProgressBar.hpp"
//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <sstream>
#include <iomanip>
#include <vector>
#include <climits>
#include <algorithm>
#include <execution>
#include <stdexcept>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cerrno> // for errno
#include <cstring> // for strerror
#include <string_view>
#include <csignal>
//--------------------------------------------------------------
// FMT library
//--------------------------------------------------------------
#ifdef HAVE_FMT
    //--------------------------
    #include <fmt/core.h>
    #include <fmt/color.h>
    #include <fmt/format.h>
    //--------------------------
#endif
//--------------------------------------------------------------
// Definitions 
//--------------------------------------------------------------
// ANSI Colors
//--------------------------
#ifdef HAVE_FMT
    //--------------------------
    #define ANSI_COLOR_BLACK        "\x1b[30m"
    #define ANSI_COLOR_RED          "\x1b[31m"
    #define ANSI_COLOR_GREEN        "\x1b[32m"
    #define ANSI_COLOR_YELLOW       "\x1b[33m"
    #define ANSI_COLOR_BLUE         "\x1b[34m"
    #define ANSI_COLOR_MAGENTA      "\x1b[35m"
    #define ANSI_COLOR_CYAN         "\x1b[36m"
    #define ANSI_COLOR_WHITE        "\x1b[37m"
    #define ANSI_COLOR_RESET        "\x1b[0m"
    //--------------------------
    // ANSI Bold
    //--------------------------
    #define ANSI_BOLD_ON            "\x1b[1m"
    #define ANSI_BOLD_OFF           ANSI_COLOR_RESET
    //--------------------------
#endif
//--------------------------
// Conversion Constant
//--------------------------
constexpr uint8_t SECONDS_PER_MINUTE        = 60;
constexpr uint16_t SECONDS_PER_HOUR         = 3600;
constexpr uint32_t SECONDS_PER_DAY          = 86400;
constexpr uint16_t MILLISECONDS_PER_SECOND  = 1000;
//--------------------------
// Boost Constant
//--------------------------
#define CIRCULAR_BUFFER         10
//--------------------------
// Interval Constant
//--------------------------
#define UPDATE_INTERVAL         5
//--------------------------
// Interval Constant
//--------------------------
#define BAR_PERCENTAGE              0.15
#define DEFAULT_WIDTH               30
#define MIN_WIDTH                   10
constexpr size_t MIN_BAR_LENGTH =   15;       // Minimum length of the progress bar for visibility.
//--------------------------------------------------------------
size_t Utils::ProgressBar::m_available_width = 0;
size_t Utils::ProgressBar::m_name_length = 0;
size_t Utils::ProgressBar::m_spaces_after_bar = 0;
size_t Utils::ProgressBar::m_bar_length = 0;
//--------------------------------------------------------------
Utils::ProgressBar::ProgressBar(const std::string& name, 
                                const std::string& progress_char, 
                                const std::string& empty_space_char) :  m_total(std::numeric_limits<size_t>::max()),
                                                                        m_progress(0),
                                                                        m_update_counter(0),
                                                                        m_name(name),
                                                                        m_progress_char(progress_char),
                                                                        m_empty_space_char(empty_space_char),
                                                                        m_delta_times(CIRCULAR_BUFFER),
                                                                        m_last_etc(std::numeric_limits<double>::max()),
                                                                        m_last_tick_time(std::chrono::steady_clock::now()),
                                                                        m_start_time(m_last_tick_time){
    //--------------------------
    m_name_length = name.length();
    //--------------------------
    calculate_bar();
    //--------------------------
    std::signal(SIGWINCH, handle_winch_signal);
    //--------------------------
}//end Utils::ProgressBar::ProgressBar
//--------------------------------------------------------------
Utils::ProgressBar::ProgressBar(const size_t& total,
                                const std::string& name, 
                                const std::string& progress_char, 
                                const std::string& empty_space_char) :  m_total(total),
                                                                        m_progress(0),
                                                                        m_update_counter(0),
                                                                        m_name(name), 
                                                                        m_progress_char(progress_char), 
                                                                        m_empty_space_char(empty_space_char), 
                                                                        m_delta_times(CIRCULAR_BUFFER),
                                                                        m_last_etc(std::numeric_limits<double>::max()),
                                                                        m_last_tick_time(std::chrono::steady_clock::now()),
                                                                        m_start_time(m_last_tick_time){
    //--------------------------
    m_name_length = name.length();
    //--------------------------
    calculate_bar();
    //--------------------------
    std::signal(SIGWINCH, handle_winch_signal);
    //--------------------------
}// end Utils::ProgressBar::ProgressBar
//--------------------------------------------------------------
void Utils::ProgressBar::update(void){
    //--------------------------
    tick();
    //--------------------------
}//end void Utils::ProgressBar::update(void)
//--------------------------------------------------------------
bool Utils::ProgressBar::done(void){
    //--------------------------
    return is_done();
    //--------------------------
}// end constexpr bool Utils::ProgressBar::done(void)
//--------------------------------------------------------------
double Utils::ProgressBar::calculate_etc(void) {
    //--------------------------
    if (m_progress <= 0 or m_total <= 0) {
        //-------------------------
        return std::numeric_limits<double>::max();
        //--------------------------
    }// end if (m_progress < 0 or m_total < 0)
    //--------------------------
    auto now = std::chrono::steady_clock::now();
    auto overall_elapsed_time = std::chrono::duration<double>(now - m_start_time).count();
    //--------------------------
    // Overall ETA calculation based on overall progress and time
    //--------------------------
    double overall_etc = overall_elapsed_time * (m_total - m_progress) / m_progress;
    //--------------------------
    if (m_update_counter == 0 or m_delta_times.size() < CIRCULAR_BUFFER) {
        //--------------------------
        // Special handling for the first few ticks
        //--------------------------
        m_last_tick_time = now; // Update the last tick time
        m_last_etc = overall_etc; // Store the calculated ETC
        //--------------------------
        return overall_etc;
        //--------------------------
    }// end if (m_update_counter == 0 or m_delta_times.size() < CIRCULAR_BUFFER)
    //--------------------------
    if (++m_update_counter % UPDATE_INTERVAL == 0) {
        //--------------------------
        auto elapsed_since_last_progress = std::chrono::duration<double>(now - m_last_tick_time).count();
        m_delta_times.push_back(elapsed_since_last_progress);
        //--------------------------
        double recent_average_time = std::reduce(std::execution::par, m_delta_times.begin(), m_delta_times.end(), 0.0) / m_delta_times.size();
        double recent_etc = recent_average_time * (m_total - m_progress) / m_progress;
        //--------------------------
        // Combine the overall ETC and recent ETC
        //--------------------------
        double combined_etc = (overall_etc + recent_etc) / 2;
        //--------------------------
        m_last_tick_time = now; // Update the last tick time
        m_last_etc = combined_etc; // Store the calculated ETC
        m_update_counter = 0; // Reset counter
        //--------------------------
        return combined_etc;
        //--------------------------
    }// end if (++m_update_counter % UPDATE_INTERVAL == 0)
    //--------------------------
    return m_last_etc; // Return last calculated ETC if not updating
    //--------------------------
}// end double Utils::ProgressBar::calculate_etc(void) 
//--------------------------------------------------------------
double Utils::ProgressBar::calculate_elapsed(void) {
    //--------------------------
    auto now = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_start_time).count();
    //--------------------------
    return elapsed_time / MILLISECONDS_PER_SECOND;  // Convert milliseconds to seconds
    //--------------------------
}// end double Utils::ProgressBar::calculate_elapsed(void)
//--------------------------------------------------------------
void Utils::ProgressBar::append_time(std::stringstream& ss, double time, const std::string& label) {
    //--------------------------
    auto days = static_cast<size_t>(time / SECONDS_PER_DAY);
    time -= days * SECONDS_PER_DAY;
    //--------------------------
    auto hours = static_cast<size_t>(time / SECONDS_PER_HOUR);
    time -= hours * SECONDS_PER_HOUR;
    //--------------------------
    auto minutes = static_cast<size_t>(time / SECONDS_PER_MINUTE);
    time -= minutes * SECONDS_PER_MINUTE;
    //--------------------------
    auto seconds = static_cast<size_t>(time);
    time -= seconds;
    //--------------------------
    auto milliseconds = static_cast<size_t>(time * MILLISECONDS_PER_SECOND);
    //--------------------------
    ss << label << " " << std::setw(2) << std::setfill('0') << days << ":"
       << std::setw(2) << std::setfill('0') << hours << ":"
       << std::setw(2) << std::setfill('0') << minutes << ":"
       << std::setw(2) << std::setfill('0') << seconds << ":"
       << std::setw(3) << std::setfill('0') << milliseconds << " ";
    //--------------------------
}// end void Utils::ProgressBar::append_time(std::stringstream& ss, double time, const std::string& label)
//--------------------------------------------------------------
#ifdef HAVE_FMT
    //--------------------------
    std::string Utils::ProgressBar::append_time(double time, const std::string& label) {
        //--------------------------
        auto days = static_cast<size_t>(time / SECONDS_PER_DAY);
        time -= days * SECONDS_PER_DAY;
        //--------------------------
        auto hours = static_cast<size_t>(time / SECONDS_PER_HOUR);
        time -= hours * SECONDS_PER_HOUR;
        //--------------------------
        auto minutes = static_cast<size_t>(time / SECONDS_PER_MINUTE);
        time -= minutes * SECONDS_PER_MINUTE;
        //--------------------------
        auto seconds = static_cast<size_t>(time);
        time -= seconds;
        //--------------------------
        auto milliseconds = static_cast<size_t>(time * MILLISECONDS_PER_SECOND);
        //--------------------------
        return fmt::format("{} {:02}:{:02}:{:02}:{:02}:{:03} ", label, days, hours, minutes, seconds, milliseconds);
        //--------------------------
    }// end std::string Utils::ProgressBar::append_time(double time, const std::string& label)
    //--------------------------
#endif
//--------------------------------------------------------------
#ifdef HAVE_FMT
    //--------------------------
    void Utils::ProgressBar::display(void) {
        //--------------------------
        size_t position = 0, percent = 0;
        //--------------------------
        if (m_total != 0 && m_total != std::numeric_limits<size_t>::max()) {
            //--------------------------
            m_progress = std::clamp(m_progress, static_cast<size_t>(0), m_total);
            //--------------------------
            auto ratio = static_cast<double>(m_progress) / m_total;
            //--------------------------
            percent = static_cast<size_t>(ratio * 100);
            //--------------------------
            position = static_cast<size_t>(m_bar_length * ratio);
            //--------------------------
        } else {
            //--------------------------
            position = m_progress % m_bar_length;
            //--------------------------
        }// end else
        //--------------------------
        std::string bar(m_bar_length, m_empty_space_char[0]);
        std::fill_n(bar.begin(), position, m_progress_char[0]);
        //--------------------------
        std::string elapsed_time = append_time(calculate_elapsed(), "Elapsed:");
        std::string etc_time = (m_total != 0 && m_total != std::numeric_limits<size_t>::max())
                            ? append_time(calculate_etc(), "ETC:") : "ETC: N/A ";
        //--------------------------
        std::string green_part = fmt::format(fmt::emphasis::bold | fmt::fg(fmt::color::green), "{}", bar.substr(0, position));
        std::string red_part = fmt::format(fmt::emphasis::bold | fmt::fg(fmt::color::red), "{}", bar.substr(position));
        //--------------------------
        std::string colored_bar = green_part + red_part + "]";
        //--------------------------
        std::string formatted_bar = fmt::format("\r{}: {:3d}% [{}{} ", m_name, percent, colored_bar, std::string(m_spaces_after_bar, ' '));
        //--------------------------
        if (m_available_width < MIN_WIDTH) {
            //--------------------------
            formatted_bar = fmt::format("\r{:3d}% [{}]", percent, bar);
        }// end if (m_available_width < 10)
        //--------------------------
        std::string formatted_time = fmt::format("{} {}", elapsed_time, etc_time);
        //--------------------------
        // Use bold emphasis
        //--------------------------
        std::cout << fmt::format(fmt::emphasis::bold, "\x1b[A{} \n{}", formatted_bar, formatted_time);
        std::cout.flush();
        //--------------------------
    }// end void Utils::ProgressBar::display(void)
//--------------------------------------------------------------
#else
//--------------------------
    void Utils::ProgressBar::display(void) {
        //--------------------------
        std::stringstream ss;
        std::string bar(m_bar_length, m_empty_space_char[0]);
        size_t position = 0, percent = 0;
        double ratio;
        //--------------------------
        ss << '\r';
        //--------------------------
        if(m_total != 0 and m_total != std::numeric_limits<size_t>::max()) {
            //--------------------------
            m_progress = std::clamp(m_progress, static_cast<size_t>(0), m_total);
            //--------------------------
            ratio = static_cast<double>(m_progress) / m_total;
            percent = static_cast<size_t>(ratio * 100);
            position = m_bar_length * ratio;
            //--------------------------
            ss << m_name << ": " << std::setw(3) << percent << "% [" << ANSI_BOLD_ON;
            //--------------------------
            std::fill_n(bar.begin(), position, m_progress_char[0]);
            //--------------------------
        } else {
            //--------------------------
            position = m_progress % m_bar_length;
            bar[position] = m_progress_char[0];
            ss << m_name << ": " << "[" << ANSI_BOLD_ON;
            //--------------------------
        }// end else
        //--------------------------
        ss << ANSI_COLOR_GREEN << bar.substr(0, position) << ANSI_COLOR_RESET;
        ss << ANSI_COLOR_RED << bar.substr(position) << ANSI_COLOR_RESET;
        ss << "] " << ANSI_BOLD_OFF;
        //--------------------------
        append_time(ss, calculate_elapsed(), "Elapsed:");
        //--------------------------
        if(m_total != 0 and m_total != std::numeric_limits<size_t>::max()) {
            //--------------------------
            append_time(ss, calculate_etc(), "ETC:");
            //--------------------------
        }// end if(m_total != 0 and m_total != std::numeric_limits<size_t>::max())
        //--------------------------
        std::cout << ss.str();
        std::cout.flush();
        //--------------------------
    }// end void ProgressBar::display(void)
    //--------------------------
#endif
//--------------------------------------------------------------
void Utils::ProgressBar::tick(void) {
    //--------------------------
    if(m_total == std::numeric_limits<size_t>::max() || !is_done()) {
        //--------------------------
        ++m_progress;
        //--------------------------
    }// end if(m_total == std::numeric_limits<size_t>::max() || !is_done())
    //--------------------------
    display();
    //--------------------------
}// end void Utils::ProgressBar::tick(void)
//--------------------------------------------------------------
bool Utils::ProgressBar::is_done(void) const{
    //--------------------------
    return m_progress >= m_total;
    //--------------------------
}// end bool Utils::ProgressBar::is_done(void)
//--------------------------------------------------------------
size_t Utils::ProgressBar::get_terminal_width(void) {
    //--------------------------
    struct winsize size;
    //--------------------------
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == -1) {
        //--------------------------
        std::cerr << "Error getting terminal size: " << std::strerror(errno) << "\n";
        return DEFAULT_WIDTH;
        //--------------------------
    }// end if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == -1)
    //--------------------------
    return size.ws_col;
    //--------------------------
}// end void Utils::ProgressBar::print_terminal_width(void)
//--------------------------------------------------------------
void Utils::ProgressBar::calculate_bar(void) {
    //--------------------------
    auto terminal_width = get_terminal_width();
    //--------------------------
    // Constants for fixed characters and escape sequences.
    //--------------------------
    constexpr size_t fixed_characters = 10;  // "100% []" and any padding.
    constexpr size_t ansi_sequences = 14;    // "\033[1m", "\033[32m", etc.
    //--------------------------
    size_t fixed_and_name_characters = m_name_length + fixed_characters;
    m_available_width = terminal_width - fixed_and_name_characters - ansi_sequences;
    //--------------------------
    // Calculate the length of the bar based on BAR_PERCENTAGE.
    //--------------------------
    m_bar_length = static_cast<size_t>(m_available_width * BAR_PERCENTAGE);
    m_bar_length = std::max(m_bar_length, MIN_BAR_LENGTH);
    m_bar_length = (m_bar_length % 2 == 0) ? m_bar_length : m_bar_length - 1;
    //--------------------------
    #ifdef HAVE_FMT
        //--------------------------
        m_spaces_after_bar = m_available_width - m_bar_length;  // Spaces to provide visual separation.
        //--------------------------
    #endif
    //--------------------------
}// end void Utils::ProgressBar::calculate_bar(void)
//--------------------------------------------------------------
void Utils::ProgressBar::handle_winch_signal(int signum) {
    //--------------------------
    if (signum == SIGWINCH) {
        //--------------------------
        calculate_bar();
        //--------------------------
    }//end if (signum == SIGWINCH)
    //--------------------------
}// end void Utils::ProgressBar::handle_winch_signal(int signum)
//--------------------------------------------------------------