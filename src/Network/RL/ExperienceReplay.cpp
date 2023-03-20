//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Network/RL/ExperienceReplay.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ExperienceReplay::sample(bool& done){
    //--------------------------
    return sample_data(done);
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ExperienceReplay::sample(bool& done)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample(const size_t& key){
    //--------------------------
    return sample_data(key);
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample(const size_t& key)
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
    m_position = ++m_position % m_capacity; //m_position += 1 % m_capacity;
    //--------------------------
    if(m_position == LONG_MAX){
        //--------------------------
        m_position = 0;
        //--------------------------
    }// end if(m_position != LONG_MAX)
    //--------------------------
}// end void ExperienceReplay::push_data(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& reward, const bool& done)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample_data(void){
    //--------------------------
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> uniform_position(0, m_memory.size()-1);
    //--------------------------
    return std::next( m_memory.begin() , uniform_position(rng))->second;
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample_data(void)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ExperienceReplay::sample_data(bool& done){
    //--------------------------
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> uniform_position(0, m_memory.size()-1);
    //--------------------------
    auto it = std::next(m_memory.begin() , uniform_position(rng));
    //--------------------------
    auto [input, next_input, reward, _done] = it->second;
    //--------------------------
    done = _done;
    //--------------------------
    return {input, next_input, reward};
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ExperienceReplay::sample_data(bool& done)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample_data(const size_t& key){
    //--------------------------
    if (key > m_memory.size()){
        //--------------------------
        throw std::out_of_range("Key: [" + std::to_string(key) + "] is larger then memory size:[" + std::to_string(m_memory.size()) + "]");
        //--------------------------
    }// end if (key > m_memory.size() or key > m_capacity)
    //--------------------------
    return m_memory.at(key);
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool> ExperienceReplay::sample_data(const size_t& key)
//--------------------------------------------------------------
size_t ExperienceReplay::map_size(void) const{
    //--------------------------
    return m_memory.size();
    //--------------------------
}// end size_t ExperienceReplay::map_size(void) const
//--------------------------------------------------------------