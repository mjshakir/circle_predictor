//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/Supervised/Normalize.hpp"
//--------------------------------------------------------------
Normalize::Normalize(const torch::Tensor& input) : m_input(input), m_max(torch::max(input)), m_min(torch::min(input)){
    //--------------------------
    if(std::abs(m_max.item().toDouble() - m_min.item().toDouble()) <= 1E-9){
        //--------------------------
        throw std::runtime_error("Divide by zero. Max:[" + std::to_string(m_max.item().toDouble()) + "] and min:[" + std::to_string(m_min.item().toDouble()) + "]");
        //--------------------------
    }// end if((m_max.item().toDouble() - m_min.item().toDouble()) == 0.)
    //--------------------------
}// end Normalize::Normalize(const torch::Tensor& input)
//--------------------------------------------------------------
torch::Tensor Normalize::normalization(void){
    //--------------------------
    return normalization_data();
    //--------------------------
}// end torch::Tensor Normalize::normalization(void)
//--------------------------------------------------------------
torch::Tensor Normalize::normalization(const torch::Tensor& input){
    //--------------------------
    return normalization_data(input);
    //--------------------------
}// end torch::Tensor Normalize::normalization(const torch::Tensor& input)
//--------------------------------------------------------------
torch::Tensor Normalize::unnormalization(const torch::Tensor& input){
    //--------------------------
    return unnormalization_data(input);
    //--------------------------
}// end torch::Tensor Normalize::unnormalization(const torch::Tensor& input)
//--------------------------------------------------------------
torch::Tensor Normalize::min(void) const{
    //--------------------------
    return get_min();
    //--------------------------
}// end torch::Tensor Normalize::get_min(void) const
//--------------------------------------------------------------
torch::Tensor Normalize::max(void) const{
    //--------------------------
    return get_max();
    //--------------------------
}// end torch::Tensor Normalize::get_max(void) const
//--------------------------------------------------------------
torch::Tensor Normalize::normalization_data(void){
    //--------------------------
    return ((m_input-m_min)/(m_max-m_min));
    //--------------------------
}// end torch::Tensor Normalize::normalization_data(void)
//--------------------------------------------------------------
torch::Tensor Normalize::normalization_vdata(const torch::Tensor& input){
    //--------------------------
    return ((input-m_min)/(m_max-m_min));
    //--------------------------
}// end torch::Tensor Normalize::normalization_data(void)
//--------------------------------------------------------------
torch::Tensor Normalize::normalization_data(const torch::Tensor& input){
    //--------------------------
    auto min_ = torch::min(input), max_ = torch::max(input);
    //--------------------------
    if((max_ - min_).item<double>() < 1E-9){
        return (input/max_);
    }// end if(min_.less_(torch::tensor(1E-9)).item<bool>())
    //--------------------------
    return ((input-min_)/(max_-min_));
    //--------------------------
}// end torch::Tensor Normalize::normalization_data(const torch::Tensor& input)
//--------------------------------------------------------------
torch::Tensor Normalize::unnormalization_data(const torch::Tensor& input){
    //--------------------------
    return (input*(m_max-m_min))+m_min;
    //--------------------------
}// end torch::Tensor Normalize::unnormalization_data(const torch::Tensor& input)
//--------------------------------------------------------------
torch::Tensor Normalize::get_min(void) const{
    //--------------------------
    return m_min;
    //--------------------------
}// end torch::Tensor Normalize::get_min(void) const
//--------------------------------------------------------------
torch::Tensor Normalize::get_max(void) const{
    //--------------------------
    return m_max;
    //--------------------------
}// end torch::Tensor Normalize::get_max(void) const
//--------------------------------------------------------------