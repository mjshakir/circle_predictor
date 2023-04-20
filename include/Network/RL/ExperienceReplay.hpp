#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
//--------------------------------------------------------------
template<typename... Args>
class ExperienceReplay{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        ExperienceReplay(void) = delete;
        //--------------------------
        ExperienceReplay(const size_t& capacity = 500) : m_capacity(capacity), m_position(0){
            //--------------------------
            m_memory.reserve(capacity);
            //--------------------------
        }// end ExperienceReplay(const size_t& capacity = 500) : m_capacity(capacity), m_position(0)
        //--------------------------
        void push(const Args&... args){
            //--------------------------
            return push_data(args...);
            //--------------------------
        }// end void void push(const Args&... args)
        //--------------------------------------------------------------
        std::tuple<Args...> sample(void){
            //--------------------------
            return sample_data();
            //--------------------------
        }// end std::tuple<Args...> sample(void)
        //--------------------------
        std::tuple<Args...> sample(const size_t& key){
            //--------------------------
            return sample_data(key);
            //--------------------------
        }// end std::tuple<Args...> sample(const size_t& key)
        //--------------------------
        size_t size(void) const{
            //--------------------------
            return map_size();
            //--------------------------
        }// end size_t size(void) const
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        void push_data(const Args&... args){
            //--------------------------
            m_memory.try_emplace(m_position, args...);
            //--------------------------
            m_position = ++m_position % m_capacity; //m_position += 1 % m_capacity;
            //--------------------------
            if(m_position == LONG_MAX){
                //--------------------------
                m_position = 0;
                //--------------------------
            }// end if(m_position != LONG_MAX)
            //--------------------------
        }// end void push_data(const Args&... args)
        //--------------------------------------------------------------
        std::tuple<Args...> sample_data(void){
            //--------------------------
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<std::mt19937::result_type> uniform_position(0, m_memory.size()-1);
            //--------------------------
            return std::next( m_memory.begin() , uniform_position(rng))->second;
            //--------------------------
        }// end std::tuple<Args...> sample_data(void)
        //--------------------------------------------------------------
        std::tuple<Args...> sample_data(const size_t& key){
            //--------------------------
            if (key > m_memory.size()){
                //--------------------------
                throw std::out_of_range("Key: [" + std::to_string(key) + "] is larger then the memory size:[" + std::to_string(m_memory.size()-1) + "]");
                //--------------------------
            }// end if (key > m_memory.size())
            //--------------------------
            return m_memory.at(key);
            //--------------------------
        }// end std::tuple<Args...> sample_data(const size_t& key)y)
        //--------------------------------------------------------------
        size_t map_size(void) const {
            //--------------------------
            return m_memory.size();
            //--------------------------
        }// end size_t map_size(void) const
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        size_t m_capacity, m_position;
        //--------------------------
        std::unordered_map<size_t, std::tuple<Args...>> m_memory;
    //--------------------------------------------------------------
}; // end class ExperienceReplay