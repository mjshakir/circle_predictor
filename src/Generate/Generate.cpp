#include "Generate/Generate.hpp"
#include <random>
#include <thread>
#include <mutex>
#include <future>
#include "Timing/Timing.hpp"
//--------------------------------------------------------------
Generate::Generate(const torch::Tensor& x_value, const double& radius, const uint16_t& generated_points) :  m_radius(radius), 
                                                                                                m_generated_points(generated_points), 
                                                                                                m_x_value(x_value){
    //--------------------------
    y_value = generate_value(x_value, radius);
    //--------------------------
}// end Generate::Generate(const torch::Tensor& x_value, const double& radius) : m_x_value(x), m_radius(radius)
//--------------------------------------------------------------
Generate::Generate(const double& radius, const uint16_t& generated_points) : m_radius(radius), m_generated_points(generated_points){
    //--------------------------
    full_data = generate_value(radius);
    //--------------------------
}// end Generate::Generate(const double& radius) : m_x_value(x), m_radius(radius)
//--------------------------------------------------------------
torch::Tensor Generate::get_x_value(void){
    //--------------------------
    return m_x_value;
    //--------------------------
}// end const double Generate::get_x_value()
//--------------------------------------------------------------
torch::Tensor Generate::get_y_value(void){
    //--------------------------
    return y_value;
    //--------------------------
}// end const double Generate::get_y_value()
//--------------------------------------------------------------
GenerateDate Generate::get_data(void){
    //--------------------------
    return full_data;
    //--------------------------
}// end const GenerateDate Generate::get_data(void)
//--------------------------------------------------------------
double Generate::get_radius(){
    //--------------------------
    return m_radius;
    //--------------------------
}// end const double Generate::get_radius()
//--------------------------------------------------------------
const torch::Tensor Generate::generate_value(const torch::Tensor& x_value, const double& radius){
    //--------------------------
    return (std::abs(radius)*sin(x_value) + std::abs(radius)*cos(x_value));
    //--------------------------
}// end const double Generate::generate_value(const double& x_value, const double& radius)
//--------------------------------------------------------------
const GenerateDate Generate::generate_value(const double& radius){
    //--------------------------
    GenerateDate data;
    //--------------------------
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
    std::uniform_real_distribution<double> uniform_angle(-1,1);
    std::default_random_engine re;
    //--------------------------
    std::vector<double> angle;
    angle.reserve(m_generated_points);
    //--------------------------
    for (size_t i = 0; i < m_generated_points; i++){
        //--------------------------
        angle.emplace_back(uniform_angle(re));
        //--------------------------
    } // end for (size_t i = 0; i < m_generated_points; i++)
    //--------------------------
    data.data = torch::tensor(angle);
    data.target = generate_value(data.data, radius);
    //--------------------------
    return data;
    //--------------------------
}// end const double Generate::generate_value(const double& x_value, const double& radius)
//--------------------------------------------------------------