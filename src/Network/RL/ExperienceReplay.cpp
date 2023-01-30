//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Network/RL/ExperienceReplay.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
#include <unordered_map>
//--------------------------------------------------------------
ExperienceReplay::ExperienceReplay(const size_t& capacity) : m_capacity(capacity), m_position(0){
    //--------------------------
    m_memory.reserve(m_capacity);
    //--------------------------
}// end ExperienceReplay::ExperienceReplay(const size_t& capacity)
//--------------------------------------------------------------
void ExperienceReplay::push(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& reward, const bool& done){
    //--------------------------
    return push_data(input, next_input, reward, done);
    //--------------------------
}// end void ExperienceReplay::push(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& reward, const bool& done)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample(void){
    //--------------------------
    return sample_data();
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample_data(void)
//--------------------------------------------------------------
size_t ExperienceReplay::size(void) const {
    //--------------------------
    return map_size();
    //--------------------------
}// end size_t ExperienceReplay::size(void) const
//--------------------------------------------------------------
void ExperienceReplay::push_data(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& reward, const bool& done){
    //--------------------------
    m_memory.try_emplace(m_position, input, next_input, reward, done);
    //--------------------------
    if(m_position != LONG_MAX){
        //--------------------------
        m_position = ++m_position % m_capacity;
        //--------------------------
    }// end if(m_position != LONG_MAX)
    //--------------------------
}// end void ExperienceReplay::push_data(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& reward, const bool& done)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample_data(void){
    //--------------------------
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
    std::uniform_int_distribution<size_t> uniform_position(0, m_memory.size());
    std::default_random_engine re;
    //--------------------------
    auto it = m_memory.begin();
    std::advance(it, uniform_position(re));
    //--------------------------
    return it->second;
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample_data(void)
//--------------------------------------------------------------
size_t ExperienceReplay::map_size(void) const{
    //--------------------------
    return m_memory.size();
    //--------------------------
}// end size_t ExperienceReplay::map_size(void) const
//--------------------------------------------------------------