//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/RL/RLGenerate.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
//-------------------
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
RLGenerate::RLGenerate(const size_t& generated_points, const size_t& column, const double& limiter) : m_limiter(limiter){
    //--------------------------
    m_data_test = generate_input(generated_points*0.2, column);
    //--------------------------
    m_data = generate_input(generated_points, column);
    //--------------------------
}// end RLGenerate::RLGenerate(const size_t& generated_points, const size_t& column, const double& limiter)
//--------------------------------------------------------------
RLGenerate::RLGenerate( const size_t& generated_points, 
                        const size_t& generated_points_test, 
                        const size_t& column, 
                        const double& limiter) : m_limiter(limiter){
    //--------------------------
    m_data_test = generate_input(generated_points_test, column);
    //--------------------------
    m_data = generate_input(generated_points, column);
    //--------------------------
}// end RLGenerate::RLGenerate( const size_t& generated_points, const size_t& generated_points_test, const size_t& column, const double& limiter)
//--------------------------------------------------------------
std::vector<torch::Tensor> RLGenerate::get_input(void){
    //--------------------------
    return m_data;
    //--------------------------
}// end torch::Tensor RLGenerate::get_input(void)
//--------------------------------------------------------------
std::vector<torch::Tensor> RLGenerate::get_test_input(void){
    //--------------------------
    return m_data_test;
    //--------------------------
}// end std::vector<torch::Tensor> RLGenerate::get_test_input(void)
//--------------------------------------------------------------
std::vector<torch::Tensor> RLGenerate::generate_input(const size_t& generated_points, const size_t& column){
    //--------------------------
    return generate_value(generated_points, column);
    //--------------------------
}// end std::vector<torch::Tensor> RLGenerate::generate_input(const size_t& generated_points, const size_t& column)
//--------------------------------------------------------------
torch::Tensor RLGenerate::get_output(const size_t& generated_points, const size_t& column){
    //--------------------------
    return generate_target(generated_points, column);
    //--------------------------
}// end std::vector<torch::Tensor> RLGenerate::get_output(const size_t& generated_points, const size_t& column)
//--------------------------------------------------------------
std::vector<torch::Tensor> RLGenerate::data(const size_t& generated_points, const size_t& column){
    //--------------------------
    return generate_value(generated_points, column);
    //--------------------------
}// end torch::Tensor RLGenerate::data(const size_t& generated_points, const size_t& column)
//--------------------------------------------------------------
std::vector<torch::Tensor> RLGenerate::generate_value(const size_t& generated_points, const size_t& column){
    //--------------------------
    // https://stackoverflow.com/questions/66396651/what-is-the-most-efficient-way-of-converting-a-stdvectorstdtuple-to-a-to
    //--------------------------
    std::vector<torch::Tensor> _data(generated_points);
    //--------------------------
    std::generate(std::execution::par, _data.begin(), _data.end(),[this, &column]() {return inner_generation(column);});
    //--------------------------
    return _data;
    //--------------------------
}// end std::vector<torch::Tensor> RLGenerate::generate_value(const size_t& generated_points, const size_t& column)
//--------------------------------------------------------------
torch::Tensor RLGenerate::inner_generation(const size_t& column){
    //--------------------------
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
    std::uniform_real_distribution<double> uniform_angle(-m_limiter, m_limiter);
    //--------------------------
    std::vector<double> _temp;
    _temp.reserve(column);
    //--------------------------
    std::vector<double> _output_data;
    _output_data.reserve(column-1);
    //--------------------------
    for (size_t i{0}; i < column-1; ++i){
        //--------------------------
        _temp.push_back(uniform_angle(gen));
        //--------------------------
        _output_data.push_back(uniform_angle(gen));
        //--------------------------
    }// end for (size_t i{0}; i < column-1; ++i)
    //--------------------------
    _temp.push_back((std::pow((_output_data.at(0) - _temp.at(0)),2) + std::pow(( _output_data.at(1) - _temp.at(1)),2)));
    //--------------------------
    return torch::tensor(_temp).view({-1,static_cast<int64_t>(column)});
    //--------------------------
}// end torch::Tensor RLGenerate::inner_generation(const size_t& column)
//--------------------------------------------------------------
torch::Tensor RLGenerate::generate_target(const size_t& generated_points, const size_t& column){
    //--------------------------
    // https://stackoverflow.com/questions/66396651/what-is-the-most-efficient-way-of-converting-a-stdvectorstdtuple-to-a-to
    //--------------------------
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
    std::uniform_real_distribution<double> uniform_angle(-m_limiter, m_limiter);
    // std::default_random_engine re;
    //--------------------------
    std::vector<double> _data(generated_points*column);
    //--------------------------
    std::generate(std::execution::par, _data.begin(), _data.end(),[&uniform_angle, &gen]() {return uniform_angle(gen);});
    //--------------------------
    return torch::tensor(_data).view({-1, static_cast<int64_t>(column)});
    //--------------------------
}// end torch::Tensor RLGenerate::generate_target(const size_t& generated_points, const size_t& column)
//--------------------------------------------------------------