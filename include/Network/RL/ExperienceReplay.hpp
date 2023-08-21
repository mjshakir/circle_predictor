#pragma once
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <iostream>
//-------------------
#include <deque>
#include <vector>
#include <tuple>
#include <random>
#include <mutex>
//--------------------------------------------------------------
template<typename... Args>
class ExperienceReplay{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        ExperienceReplay(void) = delete;
        //--------------------------
        virtual ~ExperienceReplay() = default; // Virtual destructor
        //--------------------------
        ExperienceReplay(const size_t& capacity = 500) : m_capacity(capacity){
            //--------------------------
        }// end ExperienceReplay(const size_t& capacity = 500) : m_capacity(capacity)
        //--------------------------
        ExperienceReplay(const ExperienceReplay& other) : m_capacity(other.m_capacity){
            //--------------------------
        }// end ExperienceReplay(const ExperienceReplay& other)
        //--------------------------
        ExperienceReplay& operator=(const ExperienceReplay& other) {
            //--------------------------
            // Check for self-assignment
            if (this == &other) {
                //--------------------------
                return *this;
                //--------------------------
            }// end if (this == &other)
            //--------------------------
            // Perform a deep copy of the data
            m_capacity  = other.m_capacity;
            //--------------------------
            return *this;
            //--------------------------
        }// end ExperienceReplay& operator=(const ExperienceReplay& other)
        //--------------------------
        ExperienceReplay(ExperienceReplay&&)                    = default;
        ExperienceReplay& operator=(ExperienceReplay&&)         = default;
        //--------------------------
        void push(const Args&... args){
            //--------------------------
            push_data(args...);
            //--------------------------
        }// end void void push(const Args&... args)
        //--------------------------------------------------------------
        std::tuple<Args...> sample(void){
            //--------------------------
            return sample_data();
            //--------------------------
        }// end std::tuple<Args...> sample(void)
        //--------------------------
        std::tuple<Args...> sample(const size_t& position){
            //--------------------------
            return sample_data(position);
            //--------------------------
        }// end std::tuple<Args...> sample(const size_t& position)
        //--------------------------
        std::vector<std::tuple<Args...>> samples(const size_t& samples) {
            //--------------------------
            return samples_data(samples);
            //--------------------------
        }// end  std::tuple<Args...> sample(const size_t& key)
        //--------------------------
        constexpr size_t size(void) const{
            //--------------------------
            return fifo_size();
            //--------------------------
        }// end size_t size(void) const
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        void push_data(const Args&... args){
            //--------------------------
            std::lock_guard<std::mutex> date_lock(m_mutex);
            //--------------------------
            m_memory.emplace_back(args...);
            //--------------------------
            if(m_memory.size() >= m_capacity) {
                //--------------------------
                m_memory.pop_front(); // Remove the oldest experience when capacity is reached
                //--------------------------
            }//end if(m_memory.size() >= m_capacity)
            //--------------------------
        }// end void push_data(const Args&... args)
        //--------------------------------------------------------------
        std::tuple<Args...> sample_data(void){
            //--------------------------
            thread_local std::random_device dev;
            thread_local std::mt19937 rng(dev());
            thread_local std::uniform_int_distribution<std::mt19937::result_type> uniform_position(0, m_memory.size()-1);
            //--------------------------
            return m_memory.at(uniform_position(rng));
            //--------------------------
        }// end std::tuple<Args...> sample_data(void)
        //--------------------------------------------------------------
        std::tuple<Args...> sample_data(const size_t& position){
            //--------------------------
            if (position > m_memory.size()){
                //--------------------------
                throw std::out_of_range("Position: [" + std::to_string(position) + "] is larger then the memory size:[" + std::to_string(m_memory.size()-1) + "]");
                //--------------------------
            }// end if (key > m_memory.size())
            //--------------------------
            return m_memory.at(position);
            //--------------------------
        }// end std::tuple<Args...> sample_data(const size_t& key)
        //--------------------------------------------------------------
        std::vector<std::tuple<Args...>> samples_data(const size_t& samples){
            //--------------------------
            // Check if the requested samples exceed the current memory size
            //--------------------------
            if(samples > m_memory.size()) {
                //--------------------------
                throw std::out_of_range("Samples: [" + std::to_string(samples) + "] is larger then the memory size:[" + std::to_string(m_memory.size()-1) + "]");
                //--------------------------
            }// end  if(samples > m_memory.size())
            //--------------------------
            std::vector<std::tuple<Args...>> _data;
            _data.reserve(samples);
            //--------------------------
            thread_local std::random_device dev;
            thread_local std::mt19937 rng(dev());
            //--------------------------
            std::sample(m_memory.begin(), m_memory.end(), std::back_inserter(_data), samples, rng);
            //--------------------------
            return _data;
            //--------------------------
        }// end std::tuple<Args...> samples_data(const size_t& key)
        //--------------------------------------------------------------
        constexpr size_t fifo_size(void) const {
            //--------------------------
            return m_memory.size();
            //--------------------------
        }// end size_t map_size(void) const
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        size_t m_capacity;
        //--------------------------
        std::deque<std::tuple<Args...>> m_memory;
        //--------------------------
        std::mutex m_mutex;
    //--------------------------------------------------------------
}; // end class ExperienceReplay
//--------------------------------------------------------------