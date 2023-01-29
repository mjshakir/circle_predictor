//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/RL/RLGenerate.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
//-------------------
#include <future>
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Timing/TimeIT.hpp"
//--------------------------------------------------------------
RLGenerate::RLGenerate(const size_t& generated_points, const size_t& column, const double& limiter, const torch::Device& device) : 
                        m_generated_points(generated_points),
                        m_column(column), 
                        m_limiter(limiter),
                        m_device(device){
    //--------------------------
    auto data_test_thread = std::async(std::launch::async, &RLGenerate::generate_input, this, m_generated_points*0.2, m_column);
    //--------------------------
    auto data_thread = std::async(std::launch::async, &RLGenerate::generate_input, this, m_generated_points, m_column);
    //--------------------------
    m_data_test = data_test_thread.get();
    //--------------------------
    m_data = data_thread.get();
    //--------------------------
}// end RLGenerate::RLGenerate(const size_t& generated_points)
//--------------------------------------------------------------
RLGenerate::RLGenerate( const size_t& generated_points, 
                        const size_t& generated_points_test, 
                        const size_t& column, 
                        const double& limiter, 
                        const torch::Device& device) : 
                            m_generated_points(generated_points),
                            m_column(column), 
                            m_limiter(limiter),
                            m_device(device){
    //--------------------------
    auto data_test_thread = std::async(std::launch::async, &RLGenerate::generate_input, this, generated_points_test, m_column);
    //--------------------------
    auto data_thread = std::async(std::launch::async, &RLGenerate::generate_input, this, m_generated_points, m_column);
    //--------------------------
    m_data_test = data_test_thread.get();
    //--------------------------
    m_data = data_thread.get();
    //--------------------------
}// end RLGenerate::RLGenerate(const size_t& generated_points)
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
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
    std::uniform_real_distribution<double> uniform_angle(-m_limiter, m_limiter);
    std::default_random_engine re;
    //--------------------------
    std::vector<torch::Tensor> _data;
    _data.reserve(generated_points);
    //--------------------------
    std::vector<double> _temp;
    _temp.reserve(column);
    //--------------------------
    std::vector<double> _output_data;
    _output_data.reserve(column-1);
    //--------------------------
    for (size_t i = 0; i < generated_points; ++i){
        //--------------------------
        // std::generate(std::execution::par_unseq, _temp.begin(), _temp.end(),[&uniform_angle, &re]() {return uniform_angle(re);});
        //--------------------------
        // std::generate(std::execution::par_unseq, _output_data.begin(), _output_data.end(),[&uniform_angle, &re]() {return uniform_angle(re);});
        //--------------------------
        for (size_t j = 0; j < column-1; ++j){ // end batch
            //--------------------------
            _temp.push_back(uniform_angle(re));
            //--------------------------
            _output_data.push_back(uniform_angle(re));
            //--------------------------
        }// end for (size_t i = 0; i < column-1; ++i)
        //--------------------------
        _temp.push_back((std::pow((_output_data.at(0) - _temp.at(0)),2) + std::pow(( _output_data.at(1) - _temp.at(1)),2)));
        //--------------------------
        _data.push_back(torch::tensor(_temp).view({-1,static_cast<int64_t>(column)}).to(m_device));
        //--------------------------
        _temp.clear();
        //-----------
        _output_data.clear();
        //--------------------------
    }// end for (size_t i = 0; i < generated_points; ++i)
    //--------------------------
    return _data;
    //--------------------------
}// end torch::Tensor RLGenerate::generate_input(void)
//--------------------------------------------------------------
torch::Tensor RLGenerate::generate_target(const size_t& generated_points, const size_t& column){
    //--------------------------
    // https://stackoverflow.com/questions/66396651/what-is-the-most-efficient-way-of-converting-a-stdvectorstdtuple-to-a-to
    //--------------------------
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
    std::uniform_real_distribution<double> uniform_angle(-m_limiter, m_limiter);
    std::default_random_engine re;
    //--------------------------
    std::vector<double> _data(generated_points*column);
    //--------------------------
    std::generate(std::execution::par_unseq, _data.begin(), _data.end(),[&uniform_angle, &re]() {return uniform_angle(re);});
    //--------------------------
    return torch::tensor(_data).view({-1, static_cast<int64_t>(column)}).to(m_device);
    //--------------------------
}// end std::vector<torch::Tensor> RLGenerate::generate_target(const size_t& generated_points, const size_t& column)
//--------------------------------------------------------------