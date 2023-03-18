//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Network/RL/RLNormalize.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
RLNormalize::RLNormalize(const std::vector<torch::Tensor>& input) : m_input(input){
    //--------------------------
    std::tie(m_min, m_max) = find_min_max(input);
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
    std::vector<torch::Tensor> _data = m_input;
    //--------------------------
    std::for_each(std::execution::par, _data.begin(), _data.end(), [this](auto& x){
        //--------------------------
        x = ((x-m_min)/(m_max-m_min));
        //--------------------------
    });
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
    std::vector<torch::Tensor> _data = input;
    //--------------------------
    const auto [t_min, t_max] = find_min_max(input);
    //--------------------------
    for(auto& x : _data){
        //--------------------------
        x = ((x-t_min)/(t_max-t_min));
        //--------------------------
    }// for(auto& x : input)
    //--------------------------
    return _data;
    //--------------------------
}// end torch::Tensor RLNormalize::normalization_data(const torch::Tensor& input)
//--------------------------------------------------------------
std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor>  RLNormalize::normalization_min_max_data(const std::vector<torch::Tensor>& input){
    //--------------------------
    std::vector<torch::Tensor> _data;
    _data.reserve(input.size());
    //--------------------------
    const auto [t_min, t_max] = find_min_max(input);
    //--------------------------
    for(const auto& x : input){
        //--------------------------
        _data.push_back(((x-t_min)/(t_max-t_min)));
        //--------------------------
    }// for(const auto& x : input)
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
    return (input*(t_max-t_min))+t_max;
    //--------------------------
}// end torch::Tensor RLNormalize::unnormalization_data(const torch::Tensor& input)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> RLNormalize::find_min_max(const std::vector<torch::Tensor>& input){
    //--------------------------
    auto _min_temp = std::min_element(std::execution::par, input.begin(), input.end(), 
                                    [](const auto& s1, const auto& s2){return torch::min(s1).less(torch::min(s2)).any().template item<bool>();});
    //--------------------------
    auto _max_temp = std::max_element(std::execution::par, input.begin(), input.end(), 
                                    [](const auto& s1, const auto& s2){return torch::max(s2).greater(torch::max(s1)).any().template item<bool>();});
    //--------------------------
    return {torch::min(*_min_temp), torch::max(*_max_temp)};
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor> RLNormalize::find_min_max(std::vector<torch::Tensor> input)
//--------------------------------------------------------------