#pragma once
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
#include <tuple>
#include <mutex>
#include <vector>
#include <algorithm>
//--------------------------------------------------------------
// Boost library
//--------------------------------------------------------------
#include <boost/circular_buffer.hpp>
//--------------------------------------------------------------
template<typename... Args>
class ExperienceReplayBuffer {
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        ExperienceReplayBuffer() = delete;
        //--------------------------
        virtual ~ExperienceReplayBuffer() = default; // Virtual destructor
        //--------------------------
        ExperienceReplayBuffer(const size_t& capacity = 500) : m_capacity(capacity) {
            //--------------------------
            m_memory.set_capacity(capacity);
            //--------------------------
        }//end explicit ExperienceReplayBuffer(const size_t& capacity = 500)
        //--------------------------
        ExperienceReplayBuffer(const ExperienceReplayBuffer& other) : m_capacity(other.m_capacity), m_memory(other.m_memory){
            //--------------------------
        }// end ExperienceReplay(const ExperienceReplay& other)
        //--------------------------
        ExperienceReplayBuffer& operator=(const ExperienceReplayBuffer& other) {
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
            m_memory    = other.m_memory;
            //--------------------------
            return *this;
            //--------------------------
        }// end ExperienceReplay& operator=(const ExperienceReplay& other)
        //--------------------------
        ExperienceReplayBuffer(ExperienceReplayBuffer&&)                    = default;
        ExperienceReplayBuffer& operator=(ExperienceReplayBuffer&&)         = default;
        //--------------------------
        void push(const Args&... args){
            //--------------------------
            push_data(args...);
            //--------------------------
        }// end void push(const Args&... args)
        //--------------------------
        std::tuple<Args...> sample(void) {
            //--------------------------
            return sample_data();
            //--------------------------
        }// end std::tuple<Args...> sample(void)
        //--------------------------
        std::tuple<Args...> sample(const size_t& key) {
            //--------------------------
            return sample_data(key);
            //--------------------------
        }// end  std::tuple<Args...> sample(const size_t& key)
        //--------------------------
        std::vector<std::tuple<Args...>> samples(const size_t& samples) {
            //--------------------------
            return samples_data(samples);
            //--------------------------
        }// end  std::tuple<Args...> sample(const size_t& key)
        //--------------------------
        constexpr size_t size() const {
            //--------------------------
            return buffer_size();
            //--------------------------
        }// end constexpr size_t size() const
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        void push_data(const Args&... args){
            //--------------------------
            std::lock_guard<std::mutex> data_lock(m_mutex);
            //--------------------------
            m_memory.push_back(std::make_tuple(args...));
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
        constexpr size_t buffer_size(void) const {
            //--------------------------
            return m_memory.size();
            //--------------------------
        }// end size_t buffer_size(void) const
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        size_t m_capacity;
        //--------------------------
        boost::circular_buffer<std::tuple<Args...>> m_memory;
        //--------------------------
        std::mutex m_mutex;
    //--------------------------------------------------------------
};//end ExperienceReplayBuffer
//--------------------------------------------------------------