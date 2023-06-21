//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/Supervised/Normalize.hpp"
//--------------------------------------------------------------
Normalize::Normalize(const torch::Tensor& input) : m_input(input), m_max(torch::max(input)), m_min(torch::min(input)){
    //--------------------------
}// end Normalize::Normalize(const torch::Tensor& input)
//--------------------------------------------------------------
torch::Tensor Normalize::normalization(void){
    //--------------------------
    return normalization_data();
    //--------------------------
}// end torch::Tensor Normalize::normalization(void)
//--------------------------------------------------------------
torch::Tensor Normalize::vnormalization(const torch::Tensor& input){
    //--------------------------
    return normalization_vdata(input);
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
    auto min_ = torch::min(input);
    //--------------------------
    return ((input-min_)/(torch::max(input)-min_));
    //--------------------------
}// end torch::Tensor Normalize::normalization_data(const torch::Tensor& input)
//--------------------------------------------------------------
torch::Tensor Normalize::unnormalization_data(const torch::Tensor& input){
    //--------------------------
    return (input*(m_max-m_min))+m_min;
    //--------------------------
}// end torch::Tensor Normalize::unnormalization_data(const torch::Tensor& input)
//--------------------------------------------------------------