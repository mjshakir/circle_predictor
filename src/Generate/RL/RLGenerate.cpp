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
// short form to get the previous even: x &= ~1;
RLGenerate::RLGenerate( const size_t& generated_points, 
                        const size_t& column, 
                        const double& limiter) :    m_generated_points((generated_points%2 == 0) ? generated_points : generated_points - (generated_points & 1)), 
                                                    m_generated_points_test(m_generated_points*0.2),
                                                    m_column(column),
                                                    m_limiter(limiter){
    //--------------------------
}// end RLGenerate::RLGenerate(const size_t& generated_points, const size_t& column, const double& limiter)
//--------------------------------------------------------------
RLGenerate::RLGenerate( const size_t& generated_points, 
                        const size_t& generated_points_test, 
                        const size_t& column, 
                        const double& limiter) : m_generated_points((generated_points%2 == 0) ? generated_points : generated_points - (generated_points & 1)),
                                                 m_generated_points_test((generated_points_test%2 == 0 and generated_points_test==0) ? 
                                                    generated_points_test : generated_points_test - (generated_points_test & 1)),
                                                 m_column(column),
                                                 m_limiter(limiter){
    //--------------------------
}// end RLGenerate::RLGenerate( const size_t& generated_points, const size_t& generated_points_test, const size_t& column, const double& limiter)
//--------------------------------------------------------------
std::vector<torch::Tensor> RLGenerate::get_input(void){
    //--------------------------
    return generate_input(m_generated_points, m_column);
    //--------------------------
}// end torch::Tensor RLGenerate::get_input(void)
//--------------------------------------------------------------
std::vector<torch::Tensor> RLGenerate::get_test_input(void){
    //--------------------------
    return generate_input(m_generated_points_test, m_column);
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
    std::generate_n(std::execution::par, std::inserter(_temp, _temp.begin()), column-1, [&uniform_angle, &gen](){return uniform_angle(gen);});
    //--------------------------
    std::generate_n(std::execution::par, std::inserter(_output_data, _output_data.begin()), column-1, [&uniform_angle, &gen](){return uniform_angle(gen);});
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