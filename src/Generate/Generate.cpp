#include "Generate/Generate.hpp"
#include <random>
#include "Timing/Timing.hpp"
//--------------------------------------------------------------
Generate::Generate(const torch::Tensor& x_value, const double& radius, const size_t& generated_points) :    m_radius(radius), 
                                                                                                            m_generated_points((generated_points < 20) ? 20 : generated_points), 
                                                                                                            m_x_value(x_value){
    //--------------------------
    y_value = generate_value(x_value, radius);
    //--------------------------
}// end Generate::Generate(const torch::Tensor& x_value, const double& radius, const size_t& generated_points)
//--------------------------------------------------------------
Generate::Generate(const double& radius, const size_t& generated_points) : m_radius(radius), m_generated_points((generated_points < 20) ? 20 : generated_points){
    //--------------------------
    full_data = generate_value(radius);
    test_data = generate_test_value(radius);
    //--------------------------
}// end Generate::Generate(const double& radius, const size_t& generated_points)
//--------------------------------------------------------------
torch::Tensor Generate::get_input(void){
    //--------------------------
    return m_x_value;
    //--------------------------
}// end const double Generate::get_input()
//--------------------------------------------------------------
torch::Tensor Generate::get_target(void){
    //--------------------------
    return y_value;
    //--------------------------
}// end const double Generate::get_target()
//--------------------------------------------------------------
GenerateDate Generate::get_data(void){
    //--------------------------
    return full_data;
    //--------------------------
}// end const GenerateDate Generate::get_data(void)
//--------------------------------------------------------------
GenerateDate Generate::get_test(void){
    //--------------------------
    return test_data;
    //--------------------------
}// end const GenerateDate Generate::get_test(void)
//--------------------------------------------------------------
double Generate::get_radius(void){
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
    std::uniform_real_distribution<double> uniform_angle(-2*M_PI,2*M_PI);
    std::default_random_engine re;
    //--------------------------
    std::vector<double> angle;
    angle.reserve(m_generated_points);
    //--------------------------
    for (size_t i = 0; i < m_generated_points; i++){
        //--------------------------
        angle.push_back(uniform_angle(re));
        //--------------------------
    } // end for (size_t i = 0; i < m_generated_points; i++)
    //--------------------------
    data.data = torch::tensor(angle);
    data.target = generate_value(data.data, radius);
    //--------------------------
    m_x_value = data.data;
    y_value = data.target;
    //--------------------------
    return data;
    //--------------------------
}// end const double Generate::generate_value(const double& x_value, const double& radius)
//--------------------------------------------------------------
const GenerateDate Generate::generate_test_value(const double& radius){
    //--------------------------
    GenerateDate data;
    //--------------------------
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
    std::uniform_real_distribution<double> uniform_angle(-2*M_PI,2*M_PI);
    std::default_random_engine re;
    //--------------------------
    size_t _generated_point = (m_generated_points % 2 == 0) ? (m_generated_points*0.3) : ((m_generated_points+1)*0.3);
    //--------------------------
    std::vector<double> angle;
    angle.reserve(_generated_point);
    //--------------------------
    for (size_t i = 0; i < _generated_point; i++){
        //--------------------------
        angle.push_back(uniform_angle(re));
        //--------------------------
    } // end for (size_t i = 0; i < m_generated_points; i++)
    //--------------------------
    data.data = torch::tensor(angle);
    data.target = generate_value(data.data, radius);
    //--------------------------
    return data;
    //--------------------------
}// end const double Generate::generate_value(const double& x_value, const double& radius)
//--------------------------------------------------------------4