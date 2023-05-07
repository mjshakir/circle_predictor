//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/Supervised/Normalize.hpp"
//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
Normalize::Normalize(const torch::Tensor& input) : m_input(input), m_max(torch::max(input)), m_min(torch::min(input)){
    //--------------------------
}// end Normalize::Normalize(const torch::Tensor& input)
//--------------------------------------------------------------
Normalize::Normalize(const std::vector<torch::Tensor>& input) : m_vinput(input){
    //--------------------------
    torch::Tensor min = torch::tensor(10000), max = torch::tensor(0);
    //--------------------------
    for(const auto& x : input){  
        //--------------------------
        torch::Tensor temp_min = torch::min(x), temp_max = torch::max(x);
        //--------------------------
        if(temp_min.less(min).any().item<bool>()){
            //--------------------------
            min = temp_min;
            //--------------------------
        }// end if(temp_min.less(min).any().item<bool>())
        //--------------------------
        if(temp_max.greater(max).any().item<bool>()){
            //--------------------------
            max = temp_max;
            //--------------------------
        }// end if(max.less(temp_max).any().item<bool>())
        //--------------------------
    }; // end for(const auto& x : input)
    //--------------------------
    m_min = min;
    m_max = max;
    //--------------------------
    // std::cout << "min: " << min << " max: " << max << std::endl;
    //--------------------------
}// end Normalize::Normalize(const torch::Tensor& input)
//--------------------------------------------------------------
torch::Tensor Normalize::normalization(void){
    //--------------------------
    return normalization_data();
    //--------------------------
}// end torch::Tensor Normalize::normalization(void)
//--------------------------------------------------------------
std::vector<torch::Tensor> Normalize::vnormalization(void){
    //--------------------------
    return normalization_vdata();
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
std::vector<torch::Tensor> Normalize::normalization(const std::vector<torch::Tensor>& input){
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
std::vector<torch::Tensor> Normalize::normalization_vdata(void){
    //--------------------------
    std::vector<torch::Tensor> _data;
    _data.reserve(m_vinput.size());
    //--------------------------
    for(const auto& x : m_vinput){
        //--------------------------
        _data.push_back(((x-m_min)/(m_max-m_min)));
        //--------------------------
    }// for(const auto& x : input)
    //--------------------------
    return _data;
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
    return ((input-torch::min(input))/(torch::max(input)-torch::min(input)));
    //--------------------------
}// end torch::Tensor Normalize::normalization_data(const torch::Tensor& input)
//--------------------------------------------------------------
std::vector<torch::Tensor> Normalize::normalization_data(const std::vector<torch::Tensor>& input){
    //--------------------------
    torch::Tensor min = torch::tensor(10000), max = torch::tensor(0);
    std::vector<torch::Tensor> _data;
    _data.reserve(input.size());
    //--------------------------
    for(const auto& x : input){  
        //--------------------------
        torch::Tensor temp_min = torch::min(x), temp_max = torch::max(x);
        //--------------------------
        if(temp_min.less(min).any().item<bool>()){
            //--------------------------
            min = temp_min;
            //--------------------------
        }// end if(temp_min.less(min).any().item<bool>())
        //--------------------------
        if(temp_max.greater(max).any().item<bool>()){
            //--------------------------
            max = temp_max;
            //--------------------------
        }// end if(max.less(temp_max).any().item<bool>())
        //--------------------------
    }; // end for(const auto& x : input)
    //--------------------------
    for(const auto& x : input){
        //--------------------------
        _data.push_back(((x-min)/(max-min)));
        //--------------------------
    }// for(const auto& x : input)
    //--------------------------
    return _data;
    //--------------------------
}// end torch::Tensor Normalize::normalization_data(const torch::Tensor& input)
//--------------------------------------------------------------
torch::Tensor Normalize::unnormalization_data(const torch::Tensor& input){
    //--------------------------
    return (input*(m_max-m_min))+m_min;
    //--------------------------
}// end torch::Tensor Normalize::unnormalization_data(const torch::Tensor& input)
//--------------------------------------------------------------