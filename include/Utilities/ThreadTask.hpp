#pragma once 
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <iostream>
#include <functional>
#include <mutex>
#include <any>
//--------------------------------------------------------------
namespace Utils {
    //--------------------------------------------------------------
    class ThreadTask {
        //--------------------------------------------------------------
        public:
            //--------------------------
            ThreadTask(void)                          = default;
            //--------------------------
            template <typename Func, typename... Args>
            ThreadTask(Func&& func, Args&&... args, uint8_t priority = 0u, uint8_t retries = 0u)
                            : m_function([func = std::forward<Func>(func), args...]() mutable -> std::any {
                                return std::invoke(func, args...);
                            }),
                            m_priority(priority),
                            m_retries(retries),
                            m_done(false) {
                //--------------------------
            }// end ThreadTask(Func&& func, Args&&... args, uint8_t priority = 0u, uint8_t retries = 0u)
            //--------------------------
            ThreadTask(const ThreadTask&)            = default;
            ThreadTask& operator=(const ThreadTask&) = default;
            //----------------------------
            ThreadTask(ThreadTask&&)                 = default;
            ThreadTask& operator=(ThreadTask&&)      = default;
            //--------------------------
            bool operator==(const ThreadTask& other) const {
                std::lock_guard<std::mutex> lock(m_mutex);
                return this == &other;
            }// end bool operator==(const ThreadTask& other) const
            //--------------------------
            bool operator<(const ThreadTask& other) const {
                return Comparator{}(*this, other);
            }// end bool operator==(const ThreadTask& other) const
            //--------------------------
            // Executes the task and handles retries and exceptions automatically.
            //--------------------------
            void execute(void);
            //--------------------------
            // Tries to execute the task once and returns true on success, false on failure.
            //--------------------------
            bool try_execute(void);
            //--------------------------
            std::any get_result(void) const;
            //--------------------------
            bool is_done(void) const;
            //--------------------------
            uint8_t get_retries(void) const;
            //--------------------------
            uint8_t get_priority(void) const;
            //--------------------------
            void increase_retries(const uint8_t& amount);
            //--------------------------
            void decrease_retries(const uint8_t& amount);
            //--------------------------
            void increase_retries(void);
            //--------------------------
            void decrease_retries(void);
            //--------------------------
            void increase_priority(const uint8_t& amount);
            //--------------------------
            void decrease_priority(const uint8_t& amount);
            //--------------------------
            void increase_priority(void);
            //--------------------------
            void decrease_priority(void);
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            void execute_local(void);
            //--------------------------
            bool try_execute_local(void);
            //--------------------------
            std::any get_result_local(void) const;
            //--------------------------
            bool is_done_local(void) const;
            //--------------------------
            uint8_t get_retries_local(void) const;
            //--------------------------
            uint8_t get_priority_local(void) const;
            //--------------------------
            void increase_retries_local(const uint8_t& amount);
            //--------------------------
            void decrease_retries_local(const uint8_t& amount);
            //--------------------------
            void increase_priority_local(const uint8_t& amount);
            //--------------------------
            void decrease_priority_local(const uint8_t& amount);
            //--------------------------
            struct Comparator {
                //--------------------------------------------------------------
                bool operator()(const ThreadTask& lhs, const ThreadTask& rhs) const {
                    //--------------------------
                    std::lock_guard<std::mutex> lock(lhs.m_mutex);
                    //--------------------------
                    if (lhs.m_priority != rhs.m_priority) {
                        return lhs.m_priority < rhs.m_priority;
                    }// end if (lhs.m_priority != rhs.m_priority)
                    //--------------------------
                    return lhs.m_retries < rhs.m_retries;
                    //--------------------------
                }// end bool operator()(const ThreadTask& lhs, const ThreadTask& rhs) const
                //--------------------------------------------------------------
            };// end struct Comparator
            //--------------------------------------------------------------
        private:
            std::function<std::any()> m_function;
            uint8_t m_priority, m_retries;
            bool m_done;
            std::optional<decltype(m_function())> m_result;
            mutable std::mutex m_mutex;
        //--------------------------------------------------------------
    };// end class ThreadTask
    //--------------------------------------------------------------
}//end namespace Utils
//--------------------------------------------------------------