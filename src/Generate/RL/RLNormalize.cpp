//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/RL/RLNormalize.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
RLNormalize::RLNormalize(const std::vector<torch::Tensor>& input) : m_input(input),
                                                                    m_min(find_min(input)),
                                                                    m_max(find_max(input)){
    //--------------------------
    if(std::abs(m_max.item().toDouble() - m_min.item().toDouble()) <= 1E-9){
        //--------------------------
        throw std::runtime_error("Divide by zero. Max:[" + std::to_string(m_max.item().toDouble()) + "] and min:[" + std::to_string(m_min.item().toDouble()) + "]");
        //--------------------------
    }// end if((m_max.item().toDouble() - m_min.item().toDouble()) == 0.)
    //--------------------------
}// end RLNormalize::RLNormalize(const torch::Tensor& input)
//--------------------------------------------------------------
std::vector<torch::Tensor> RLNormalize::normalization(void){
    //--------------------------
    return normalization_data();
    //--------------------------
}// end torch::Tensor RLNormalize::normalization(void)
//--------------------------------------------------------------
torch::Tensor RLNormalize::normalization(const torch::Tensor& input){
    //--------------------------
    return normalization_data(input);
    //--------------------------
}// end torch::Tensor RLNormalize::normalization(void)
//--------------------------------------------------------------
std::vector<torch::Tensor> RLNormalize::normalization(const std::vector<torch::Tensor>& input){
    //--------------------------
    return normalization_data(input);
    //--------------------------
}// end torch::Tensor RLNormalize::normalization(const torch::Tensor& input)
//--------------------------------------------------------------
std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor> RLNormalize::normalization_min_max(const std::vector<torch::Tensor>& input){
    //--------------------------
    return normalization_min_max_data(input);
    //--------------------------
}// std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor> RLNormalize::normalization_min_max(const std::vector<torch::Tensor>& input)
//--------------------------------------------------------------
torch::Tensor RLNormalize::unnormalization(const torch::Tensor& input){
    //--------------------------
    return unnormalization_data(input);
    //--------------------------
}// end torch::Tensor RLNormalize::unnormalization(const torch::Tensor& input)
//--------------------------------------------------------------
torch::Tensor RLNormalize::unnormalization(const torch::Tensor& input, const torch::Tensor& t_min, const torch::Tensor& t_max){
    //--------------------------
    return unnormalization_data(input, t_min, t_max);
    //--------------------------
}// end torch::Tensor RLNormalize::unnormalization(const torch::Tensor& input, const torch::Tensor min, const torch::Tensor max)
//--------------------------------------------------------------
std::vector<torch::Tensor> RLNormalize::normalization_data(void){
    //--------------------------
    std::vector<torch::Tensor> _data(m_input.size());
    //--------------------------
    std::transform(std::execution::par, m_input.begin(), m_input.end(), _data.begin(), [this](const auto& x){return (x-m_min)/(m_max-m_min);});
    //--------------------------
    return _data;
    //--------------------------
}// end torch::Tensor RLNormalize::normalization_data(void)
//--------------------------------------------------------------
torch::Tensor RLNormalize::normalization_data(const torch::Tensor& input){
    //--------------------------
    return ((input-m_min)/(m_max-m_min));
    //--------------------------
}// end torch::Tensor RLNormalize::normalization_data(void)
//--------------------------------------------------------------
std::vector<torch::Tensor> RLNormalize::normalization_data(const std::vector<torch::Tensor>& input){
    //--------------------------
    std::vector<torch::Tensor> _data(input.size());
    //--------------------------
    torch::Tensor t_min, t_max; 
    std::tie(t_min, t_max) = find_min_max(input);
    //--------------------------
    std::transform(std::execution::par, input.begin(), input.end(), _data.begin(), [&t_min, &t_max](const auto& x){return (x-t_min)/(t_max-t_min);});
    //--------------------------
    return _data;
    //--------------------------
}// end std::vector<torch::Tensor> RLNormalize::normalization_data(const std::vector<torch::Tensor>& input)
//--------------------------------------------------------------
std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor>  RLNormalize::normalization_min_max_data(const std::vector<torch::Tensor>& input){
    //--------------------------
    std::vector<torch::Tensor> _data(input.size());
    //--------------------------
    torch::Tensor t_min, t_max; 
    std::tie(t_min, t_max) = find_min_max(input);
    //--------------------------
    std::transform(std::execution::par, input.begin(), input.end(), _data.begin(), [&t_min, &t_max](const auto& x){return (x-t_min)/(t_max-t_min);});
    //--------------------------
    return {_data, t_min, t_max};
    //--------------------------
}// end std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor>  RLNormalize::normalization_min_max_data(const std::vector<torch::Tensor>& input)
//--------------------------------------------------------------
torch::Tensor RLNormalize::unnormalization_data(const torch::Tensor& input){
    //--------------------------
    return (input*(m_max-m_min))+m_min;
    //--------------------------
}// end torch::Tensor RLNormalize::unnormalization_data(const torch::Tensor& input)
//--------------------------------------------------------------
torch::Tensor RLNormalize::unnormalization_data(const torch::Tensor& input, const torch::Tensor& t_min, const torch::Tensor& t_max){
    //--------------------------
    return (input*(t_max-t_min))+t_min;
    //--------------------------
}// end torch::Tensor RLNormalize::unnormalization_data(const torch::Tensor& input)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> RLNormalize::find_min_max(const std::vector<torch::Tensor>& input){
    //--------------------------
    return {torch::min(*std::min_element(std::execution::par, input.begin(), input.end(), 
                                    [](const auto& first, const auto& second){return torch::min(first).less(torch::min(second)).any().template item<bool>();})), 
            torch::max(*std::max_element(std::execution::par, input.begin(), input.end(), 
                                    [](const auto& first, const auto& second){return torch::max(second).greater(torch::max(first)).any().template item<bool>();}))};
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor> RLNormalize::find_min_max(std::vector<torch::Tensor> input)
//--------------------------------------------------------------
torch::Tensor RLNormalize::find_min(const std::vector<torch::Tensor>& input){
    //--------------------------
    return torch::min(*std::min_element(std::execution::par, input.begin(), input.end(), 
                                    [](const auto& first, const auto& second){return torch::min(first).less(torch::min(second)).any().template item<bool>();}));
    //--------------------------
}// end torch::Tensor RLNormalize::find_min(const std::vector<torch::Tensor>& input)
//--------------------------------------------------------------
torch::Tensor RLNormalize::find_max(const std::vector<torch::Tensor>& input){
    //--------------------------
    return torch::max(*std::max_element(std::execution::par, input.begin(), input.end(), 
                                    [](const auto& first, const auto& second){return torch::max(second).greater(torch::max(first)).any().template item<bool>();}));
    //--------------------------
}// end torch::Tensor RLNormalize::find_max(const std::vector<torch::Tensor>& input)
//--------------------------------------------------------------
