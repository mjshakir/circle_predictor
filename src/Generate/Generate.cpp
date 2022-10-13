//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/Generate.hpp"
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include <random>
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Timing/Timing.hpp"
//--------------------------------------------------------------
Generate::Generate(const torch::Tensor& x_value, const double& radius, const size_t& generated_points, const std::tuple<double, double>& center) 
                    :   m_radius(radius), 
                        m_generated_points((generated_points < 200) ? 200 : generated_points), 
                        m_x_value(x_value),
                        m_center(center){
    //--------------------------
    y_value = generate_value(x_value);
    //--------------------------
}// end Generate::Generate(const torch::Tensor& x_value, const double& radius, const size_t& generated_points)
//--------------------------------------------------------------
Generate::Generate(const double& radius, const size_t& generated_points, const std::tuple<double, double>& center) 
                    :   m_radius(radius), 
                        m_generated_points((generated_points < 200) ? 200 : generated_points),
                        m_center(center){
    //--------------------------
    full_data = generate_value();
    validation_data = generate_validation_value();
    //--------------------------
}// end Generate::Generate(const double& radius, const size_t& generated_points)
//--------------------------------------------------------------
Generate::Generate(double&& radius, size_t&& generated_points, std::tuple<double, double>&& center) 
                    :   m_radius(std::move(radius)), 
                        m_generated_points((generated_points < 200) ? 200 : std::move(generated_points)),
                        m_center(std::move(center)){
    //--------------------------
    full_data = generate_value();
    validation_data = generate_validation_value();
    //--------------------------
}// end Generate::Generate(const double& radius, const size_t& generated_points)
//--------------------------------------------------------------
torch::Tensor Generate::get_input(void){
    //--------------------------
    return m_x_value;
    //--------------------------
}// end const torch::Tensor Generate::get_input()
//--------------------------------------------------------------
torch::Tensor Generate::get_target(void){
    //--------------------------
    return y_value;
    //--------------------------
}// end const torch::Tensor Generate::get_target()
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> Generate::get_data(void){
    //--------------------------
    return full_data;
    //--------------------------
}// end const std::tuple<torch::Tensor, torch::Tensor> Generate::get_data(void)
//--------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> Generate::get_validation(void){
    //--------------------------
    return validation_data;
    //--------------------------
}// end const std::tuple<torch::Tensor, torch::Tensor> Generate::get_validation(void)
//--------------------------------------------------------------
std::tuple<double, double> Generate::get_center(void){
    //--------------------------
    return m_center;
    //--------------------------
}// end const std::tuple<int16_t, int16_t> Generate::get_center(void)
//--------------------------------------------------------------
double Generate::get_radius(void){
    //--------------------------
    return m_radius;
    //--------------------------
}// end const double Generate::get_radius()
//--------------------------------------------------------------
const torch::Tensor Generate::generate_value(const torch::Tensor& x_value){
    //--------------------------
    // return (std::abs(radius)*sin(x_value) + std::abs(radius)*cos(x_value));
    //--------------------------
    return torch::sqrt(std::pow(m_radius, 2) - torch::pow((x_value - std::get<0>(m_center)),2)) + std::get<1>(m_center);
    //--------------------------
}// end const torch::Tensor Generate::generate_value(const double& x_value, const double& radius)
//--------------------------------------------------------------
const std::vector<double> Generate::generate_value(const std::vector<double>& x_value){
    //--------------------------
    std::vector<double> _target;
    _target.reserve(x_value.size());
    //--------------------------
    for (const auto& x : x_value){
        //--------------------------
        double _inner_sqrt = std::pow(m_radius, 2) - std::pow((x - std::get<0>(m_center)), 2);
        // std::cout << "_inner_sqrt: " << _inner_sqrt << std::endl;
        //--------------------------
        if(_inner_sqrt >= 0){
            //--------------------------
            _target.push_back(std::sqrt(_inner_sqrt) + std::get<1>(m_center));
            //--------------------------
        }// end if(_inner_sqrt >=0)
        else{
            //--------------------------
            _target.push_back(-std::sqrt(std::abs(_inner_sqrt)) + std::get<1>(m_center));
            //--------------------------
        }// end else
    }// end for (const auto x : x_value)
    //--------------------------
    return _target;
    //--------------------------
}// end const torch::Tensor Generate::generate_value(const double& x_value, const double& radius)
//--------------------------------------------------------------
const std::tuple<torch::Tensor, torch::Tensor> Generate::generate_value(void){
    //--------------------------
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
    std::uniform_real_distribution<double> uniform_angle(-m_radius, m_radius);
    std::default_random_engine re;
    //--------------------------
    std::vector<double> x_location;
    x_location.reserve(m_generated_points);
    //--------------------------
    for (size_t i = 0; i < m_generated_points; ++i){
        //--------------------------
        x_location.push_back((std::get<0>(m_center)+uniform_angle(re)));
        //--------------------------
    } // end for (size_t i = 0; i < m_generated_points; ++i)
    //--------------------------
    m_x_value = torch::tensor(x_location);
    auto _target = generate_value(x_location, m_radius);
    y_value = torch::transpose(torch::cat({torch::tensor(_target), (2*std::get<1>(m_center))-torch::tensor(_target)}).view({2,-1}), 0, 1);
    //--------------------------
    // std::cout << "m_x_value: \n" << m_x_value << " y_value: \n" << y_value << std::endl;
    //--------------------------
    return {m_x_value, y_value};
    //--------------------------
}// end const std::tuple<torch::Tensor, torch::Tensor> Generate::generate_value(const double& x_value, const double& radius)
//--------------------------------------------------------------
const std::tuple<torch::Tensor, torch::Tensor> Generate::generate_validation_value(void){
    //--------------------------
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
    std::uniform_real_distribution<double> uniform_angle(-m_radius, m_radius);
    std::default_random_engine re;
    //--------------------------
    size_t _generated_point = (m_generated_points % 2 == 0) ? (m_generated_points*0.3) : ((m_generated_points+1)*0.3);
    //--------------------------
    std::vector<double> x_location;
    x_location.reserve(_generated_point);
    //--------------------------
    for (size_t i = 0; i < _generated_point; ++i){
        //--------------------------
        x_location.push_back((std::get<0>(m_center)+uniform_angle(re)));
        //--------------------------
    } // end for (size_t i = 0; i < _generated_point; ++i)
    //--------------------------
    auto _test_input = torch::tensor(x_location);
    //--------------------------
    auto _target = generate_value(x_location, m_radius);
    auto _test_target = torch::transpose(torch::cat({torch::tensor(_target), (2*std::get<1>(m_center))-torch::tensor(_target)}).view({2,-1}), 0, 1);
    //--------------------------
    return {_test_input, _test_target};
    //--------------------------
}// end const std::tuple<torch::Tensor, torch::Tensor> Generate::generate_value(const double& x_value, const double& radius)
//--------------------------------------------------------------