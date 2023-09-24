#pragma once 
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <iostream>
#include <functional>
#include <mutex>
#include <any>
#include <condition_variable>
#include <future>
#include <tuple>
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
            ThreadTask(Func&& func, Args&&... args, uint8_t priority = 0U, uint8_t retries = 0U)
                            :   m_priority(priority),
                                m_retries(retries),
                                m_state(TaskState::PENDING){
                
                auto shared_func = std::make_shared<Func>(std::move(func));

                m_function = [shared_func, ...capturedArgs = std::forward<Args>(args)]() mutable -> std::any {
                    if constexpr (is_void_function<Func>()) {
                        std::invoke(*shared_func, std::forward<decltype(capturedArgs)>(capturedArgs)...);
                        return std::any{};
                    } else {
                        return std::invoke(*shared_func, std::forward<decltype(capturedArgs)>(capturedArgs)...);
                    }
                };


            }// end ThreadTask(Func&& func, Args&&... args, uint8_t priority = 0u, uint8_t retries = 0u)
            //--------------------------
            // ThreadTask(const ThreadTask&)            = default;
            // ThreadTask& operator=(const ThreadTask&) = default;
            //--------------------------------------------------------------
            ThreadTask(ThreadTask&& other) noexcept;         
            //--------------------------
            ThreadTask& operator=(ThreadTask&& other) noexcept;
            //--------------------------------------------------------------
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
            std::future<std::any> get_future(void);
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
            //--------------------------
            uint8_t get_status(void) const;
            //--------------------------
            template<typename Func>
            static constexpr bool is_void_function(void) {
                return std::is_same_v<std::invoke_result_t<Func>, void>;
            }// end constexpr bool is_void_function(void)
            //--------------------------
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            void execute_local(void);
            //--------------------------
            bool try_execute_local(void);
            //--------------------------
            template<typename Func>
            std::enable_if_t<!is_void_function<Func>(), std::future<std::any>>
            get_future_local() {
                std::lock_guard<std::mutex> lock(m_mutex);

                static_assert(!is_void_function<Func>(), "Cannot get future from a void function!");

                if (m_state == TaskState::RETRIEVED) {
                    throw std::logic_error("Future already retrieved!");
                }
                
                if (m_state == TaskState::PENDING) {
                    throw std::logic_error("Task not yet executed!");
                }
                
                m_state = TaskState::RETRIEVED;
                return m_promise.get_future();
            }
            //--------------------------
            template<typename Func>
            std::enable_if_t<std::is_void_v<Func>>
            get_future_local() {
                static_assert(is_void_function<Func>(), "Cannot get future from a void function!");
            }
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
                    std::scoped_lock lock(lhs.m_mutex, rhs.m_mutex);
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
            //--------------------------------------------------------------
            std::function<std::any()> m_function;
            uint8_t m_priority, m_retries;
            //--------------------------
            enum class TaskState : uint8_t {
                PENDING = 0,
                COMPLETED,
                RETRIEVED
            };
            //--------------------------
            TaskState m_state;
            //--------------------------
            std::promise<std::any> m_promise;
            mutable std::mutex m_mutex;
        //--------------------------------------------------------------
    };// end class ThreadTask
    //--------------------------------------------------------------
}//end namespace Utils
//--------------------------------------------------------------