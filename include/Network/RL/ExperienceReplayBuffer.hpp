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
        explicit ExperienceReplayBuffer(const size_t& capacity = 500) : m_capacity(capacity),  m_rng{m_rd()} {
            //--------------------------
            m_memory.set_capacity(capacity);
            //--------------------------
        }//end explicit ExperienceReplayBuffer(const size_t& capacity = 500)
        //--------------------------
        ExperienceReplayBuffer(const ExperienceReplayBuffer& other)             = default;
        ExperienceReplayBuffer& operator=(const ExperienceReplayBuffer& other)  = default;
        //--------------------------
        ExperienceReplayBuffer(ExperienceReplayBuffer&&)             = delete;
        ExperienceReplayBuffer& operator=(ExperienceReplayBuffer&&)  = delete;
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
        virtual ~ExperienceReplayBuffer() = default; // Virtual destructor
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
            thread_local std::uniform_int_distribution<std::mt19937::result_type> uniform_position(0, m_memory.size()-1);
            //--------------------------
            return m_memory.at(uniform_position(m_rng));
            //--------------------------
        }// end std::tuple<Args...> sample_data(void)
        //--------------------------------------------------------------
        std::tuple<Args...> sample_data(const size_t& position){
            //--------------------------
            if (key > m_memory.size()){
                //--------------------------
                throw std::out_of_range("Position: [" + std::to_string(position) + "] is larger then the memory size:[" + std::to_string(m_memory.size()-1) + "]");
                //--------------------------
            }// end if (key > m_memory.size())
            //--------------------------
            return m_memory.at(key);
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
            std::vector<size_t> sampled_indices(samples);
            //--------------------------
            std::sample(m_memory.begin(), m_memory.end(), sampled_indices.begin(), samples, m_rng);
            //--------------------------
            for(const auto& idx : sampled_indices) {
                //--------------------------
                _data.push_back(m_memory.at(idx));
                //--------------------------
            }// end for(const auto& idx : sampled_indices)
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
        std::random_device m_rd;
        std::mt19937 m_rng;
        //--------------------------
        boost::circular_buffer<std::tuple<Args...>> m_memory;
        //--------------------------
        std::mutex m_mutex;
    //--------------------------------------------------------------
};//end ExperienceReplayBuffer
//--------------------------------------------------------------