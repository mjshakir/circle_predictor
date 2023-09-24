#pragma once 
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <functional>
#include <optional>
#include <mutex>
//--------------------------------------------------------------
namespace Utils{
    //--------------------------------------------------------------
    template<typename T, typename Container = std::vector<T>, typename Comparator = std::less<typename Container::value_type>>
    class PriorityQueue {
        //--------------------------------------------------------------
        public:
            PriorityQueue(void)                            = default;
            //--------------------------
            PriorityQueue(const PriorityQueue&)            = default;
            PriorityQueue& operator=(const PriorityQueue&) = default;
            //----------------------------
            PriorityQueue(PriorityQueue&&)                 = default;
            PriorityQueue& operator=(PriorityQueue&&)      = default;
            //--------------------------
            bool empty(void) const {
                return is_empty();
            }// end bool empty(void) const
            //--------------------------
            void reserve(const size_t& size) {
                reserve_m_data(size);
            }// endvoid reserve(const size_t& size)
            //--------------------------
            void push(const T& value){
                push_internal(value);
            }// end void push(const T& value)
            //--------------------------
            void push(T&& value){
                push_internal(std::move(value));
            }// end void push(const T& value)
            //--------------------------
            template<typename... Args>
            void emplace(Args&&... args) {
                emplace_internal(std::forward<Args>(args)...);
            }// end void emplace(Args&&... args)
            //--------------------------
            std::optional<T> top(void) const {
                return top_internal();
            }// end std::optional<T> top() const
            //--------------------------
            void pop(void) {
                pop_internal();
            }// end void pop(void)
            //--------------------------
            std::optional<T> pop_top(void) {
                return pop_top_internal();
            }// end std::optional<T> pop_top(void)
            //--------------------------
            size_t size(void) const {
                return size_local();
            }// end constexpr size_t size(void)
            //--------------------------
            void remove(void) {
                remove_tasks();
            }// end void remove(void)
            //--------------------------
            void remove(const T& value){
                remove_task(value);
            }// end void remove(const T& value)
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            bool is_empty(void) const {
                std::lock_guard<std::mutex> lock(m_mutex);
                return m_data.empty();
            }// end bool is_empty() const
            //--------------------------
            void reserve_m_data(const size_t& size) {
                //--------------------------
                std::lock_guard<std::mutex> lock(m_mutex);
                m_data.reserve(size);
                //--------------------------
            }// end void reserve_m_data(const size_t& size)
            //--------------------------
            void push_internal(const T& value) {
                //--------------------------
                std::lock_guard<std::mutex> lock(m_mutex);
                m_data.push_back(value);
                std::push_heap(m_data.begin(), m_data.end(), m_comp);
                //--------------------------
            }// end void push_internal(const T& value)
            //--------------------------
            void push_internal(T&& value) {
                //--------------------------
                std::lock_guard<std::mutex> lock(m_mutex);
                m_data.push_back(std::move(value));
                std::push_heap(m_data.begin(), m_data.end(), m_comp);
                //--------------------------
            }// end void push_internal(const T& value)
            //--------------------------
            template<typename... Args>
            void emplace_internal(Args&&... args) {
                //--------------------------
                std::lock_guard<std::mutex> lock(m_mutex);
                m_data.emplace_back(std::forward<Args>(args)...);
                std::push_heap(m_data.begin(), m_data.end(), m_comp);
                //--------------------------
            }// end void emplace_internal(Args&&... args)
            //--------------------------
            std::optional<T> top_internal(void) const {
                //--------------------------
                std::lock_guard<std::mutex> lock(m_mutex);
                //--------------------------
                if (m_data.empty()) {
                    return std::nullopt;
                }// end if (m_data.empty())
                //--------------------------
                return m_data.front();
                //--------------------------
            }// end std::optional<T> top_internal(void) const
            //--------------------------
            void pop_internal(void) {
                //--------------------------
                std::lock_guard<std::mutex> lock(m_mutex);
                //--------------------------
                if (!m_data.empty()) {
                    //--------------------------
                    std::pop_heap(m_data.begin(), m_data.end(), m_comp);
                    m_data.pop_back();
                    //--------------------------
                }// end if (!m_data.empty())
                //--------------------------   
            }// end void pop_internal(void)
            //--------------------------
            std::optional<T> pop_top_internal(void) {
                //--------------------------
                std::lock_guard<std::mutex> lock(m_mutex);
                //--------------------------
                if (m_data.empty()) {
                    return std::nullopt;
                }// end if (m_data.empty())
                //--------------------------
                T top_element = std::move(m_data.front());
                std::pop_heap(m_data.begin(), m_data.end(), m_comp);
                m_data.pop_back();
                return top_element;
                //--------------------------
            }// end std::optional<T> pop_top_internal(void)
            //--------------------------
            size_t size_local(void) const{
                //--------------------------
                std::lock_guard<std::mutex> lock(m_mutex);
                //--------------------------
                return m_data.size();
                //--------------------------
            }// end constexpr size_t size(void)
            //--------------------------
            void remove_tasks(void) {
                //--------------------------
                std::lock_guard<std::mutex> lock(m_mutex);
                //--------------------------
                auto partition_point = std::partition(m_data.begin(), m_data.end(), [](const T& task) {
                    return !task.is_done();
                });
                //--------------------------
                m_data.erase(partition_point, m_data.end());
                std::make_heap(m_data.begin(), m_data.end(), m_comp);
                //--------------------------
            }// end void remove_done_tasks(void)
            //--------------------------
            void remove_task(const T& value) {
                //--------------------------
                std::lock_guard<std::mutex> lock(m_mutex);
                //--------------------------
                auto new_end = std::remove(m_data.begin(), m_data.end(), value);
                m_data.erase(new_end, m_data.end());
                std::make_heap(m_data.begin(), m_data.end(), m_comp);
                //--------------------------
            }// end void remove_task(const T& value)
            //--------------------------------------------------------------
        private:
            //--------------------------------------------------------------
            Container m_data;
            Comparator m_comp;
            mutable std::mutex m_mutex;
        //--------------------------------------------------------------
    };// end class PriorityQueue 
    //--------------------------------------------------------------
}//end namespace Utils
//--------------------------------------------------------------