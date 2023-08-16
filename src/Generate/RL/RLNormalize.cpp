//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/RL/RLNormalize.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <algorithm>
#include <execution>
#include <future>
//--------------------------------------------------------------
RL::RLNormalize::RLNormalize(const std::vector<torch::Tensor>& input) : Normalize(torch::cat(input, 0)),
                                                                        m_input(input){
    //--------------------------
}// end RL::RLNormalize::RLNormalize(const torch::Tensor& input)
//--------------------------------------------------------------
std::vector<torch::Tensor> RL::RLNormalize::normalization(void){
    //--------------------------
    return normalization_data();
    //--------------------------
}// end torch::Tensor RL::RLNormalize::normalization(void)
//--------------------------------------------------------------
torch::Tensor RL::RLNormalize::normalization(const torch::Tensor& input){
    //--------------------------
    return normalization_vdata(input);
    //--------------------------
}// end torch::Tensor RL::RLNormalize::normalization(void)
//--------------------------------------------------------------
std::vector<torch::Tensor> RL::RLNormalize::normalization(const std::vector<torch::Tensor>& input){
    //--------------------------
    return normalization_data(input);
    //--------------------------
}// end torch::Tensor RL::RLNormalize::normalization(const torch::Tensor& input)
//--------------------------------------------------------------
std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor> RL::RLNormalize::normalization_min_max(const std::vector<torch::Tensor>& input){
    //--------------------------
    return normalization_min_max_data(input);
    //--------------------------
}// std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor> RL::RLNormalize::normalization_min_max(const std::vector<torch::Tensor>& input)
//--------------------------------------------------------------
torch::Tensor RL::RLNormalize::unnormalization(const torch::Tensor& input, const torch::Tensor& t_min, const torch::Tensor& t_max){
    //--------------------------
    return unnormalization_data(input, t_min, t_max);
    //--------------------------
}// end torch::Tensor RL::RLNormalize::unnormalization(const torch::Tensor& input, const torch::Tensor min, const torch::Tensor max)
//--------------------------------------------------------------
std::vector<torch::Tensor> RL::RLNormalize::normalization_data(void){
    //--------------------------
    std::vector<torch::Tensor> _data(m_input.size());
    //--------------------------
    std::transform(std::execution::par, m_input.begin(), m_input.end(), _data.begin(), [this](const auto& x){return (x-this->get_min())/(this->get_max()-this->get_min());});
    //--------------------------
    return _data;
    //--------------------------
}// end torch::Tensor RL::RLNormalize::normalization_data(void)
//--------------------------------------------------------------
std::vector<torch::Tensor> RL::RLNormalize::normalization_data(const std::vector<torch::Tensor>& input){
    //--------------------------
    std::vector<torch::Tensor> _data(input.size());
    //--------------------------
    torch::Tensor _input_tensor = torch::cat(input, 0), t_min = torch::min(_input_tensor), t_max = torch::max(_input_tensor); 
    //--------------------------
    std::transform(std::execution::par, input.begin(), input.end(), _data.begin(), [&t_min, &t_max](const auto& x){return (x-t_min)/(t_max-t_min);});
    //--------------------------
    return _data;
    //--------------------------
}// end std::vector<torch::Tensor> RL::RLNormalize::normalization_data(const std::vector<torch::Tensor>& input)
//--------------------------------------------------------------
std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor>  RL::RLNormalize::normalization_min_max_data(const std::vector<torch::Tensor>& input){
    //--------------------------
    std::vector<torch::Tensor> _data(input.size());
    //--------------------------
    torch::Tensor _input_tensor = torch::cat(input, 0), t_min = torch::min(_input_tensor), t_max = torch::max(_input_tensor);
    //--------------------------
    std::transform(std::execution::par, input.begin(), input.end(), _data.begin(), [&t_min, &t_max](const auto& x){return (x-t_min)/(t_max-t_min);});
    //--------------------------
    return {_data, t_min, t_max};
    //--------------------------
}// end std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor>  RL::RLNormalize::normalization_min_max_data(const std::vector<torch::Tensor>& input)
//--------------------------------------------------------------
torch::Tensor RL::RLNormalize::unnormalization_data(const torch::Tensor& input, const torch::Tensor& t_min, const torch::Tensor& t_max){
    //--------------------------
    return (input*(t_max-t_min))+t_min;
    //--------------------------
}// end torch::Tensor RL::RLNormalize::unnormalization_data(const torch::Tensor& input)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> RL::RLNormalize::find_min_max(const std::vector<torch::Tensor>& input){
    //--------------------------
    auto min_thread = std::async(std::launch::async, &RLNormalize::find_min, this, input);
    auto max_thread = std::async(std::launch::async, &RLNormalize::find_max, this, input);
    //--------------------------
    return {min_thread.get(), max_thread.get()};
    //--------------------------
}// end std::tuple<torch::Tensor, torch::Tensor> RL::RLNormalize::find_min_max(std::vector<torch::Tensor> input)
//--------------------------------------------------------------
torch::Tensor RL::RLNormalize::find_min(const std::vector<torch::Tensor>& input){
    //--------------------------
    return torch::min(*std::min_element(std::execution::par, input.begin(), input.end(), 
                                    [](const auto& first, const auto& second){return torch::min(first).less(torch::min(second)).any().template item<bool>();}));
    //--------------------------
}// end torch::Tensor RL::RLNormalize::find_min(const std::vector<torch::Tensor>& input)
//--------------------------------------------------------------
torch::Tensor RL::RLNormalize::find_max(const std::vector<torch::Tensor>& input){
    //--------------------------
    return torch::max(*std::max_element(std::execution::par, input.begin(), input.end(), 
                                    [](const auto& first, const auto& second){return torch::max(second).greater(torch::max(first)).any().template item<bool>();}));
    //--------------------------
}// end torch::Tensor RL::RLNormalize::find_max(const std::vector<torch::Tensor>& input)
//--------------------------------------------------------------