//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/GenerateRL.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
#include <iostream>
//--------------------------------------------------------------
GenerateRL::GenerateRL(const size_t& generated_points, const double& limiter) : m_generated_points(generated_points), m_limiter(limiter) {
    //--------------------------
    m_data = generate_value();
    //--------------------------
}// end GenerateRL::GenerateRL(const size_t& generated_points)
//--------------------------------------------------------------
torch::Tensor GenerateRL::get_data(void){
    //--------------------------
    return m_data;
    //--------------------------
}// end torch::Tensor GenerateRL::get_data(void)
//--------------------------------------------------------------
torch::Tensor GenerateRL::generate_value(void) const{
    //--------------------------
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
    std::uniform_real_distribution<double> uniform_angle(-m_limiter, m_limiter);
    std::default_random_engine re;
    //--------------------------
    std::vector<std::tuple<double, std::tuple<double, double>>> _data;
    _data.reserve(m_generated_points);
    //--------------------------
    for (size_t i = 0; i < m_generated_points; ++i){
        //--------------------------
        _data.emplace_back({uniform_angle(re), {uniform_angle(re), uniform_angle(re)}});
        //--------------------------
    }// end for (size_t i = 0; i < m_generated_points; i++)
    //--------------------------
    return torch::tensor(_data);
    //--------------------------
}// end torch::Tensor GenerateRL::generate_value(void)
//--------------------------------------------------------------