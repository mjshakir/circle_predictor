//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/RL/GeneratePoints.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
// short form to get the previous even: x &= ~1;
RL::GeneratePoints::GeneratePoints( const size_t& generated_points, 
                                    const size_t& column, 
                                    const double& limiter) :    Generate(generated_points),
                                                                m_column(column),
                                                                m_gen(m_rd()),
                                                                m_random_point(-limiter, limiter){
    //--------------------------
}// end RL::GeneratePoints::GeneratePoints(const size_t& generated_points, const size_t& column, const double& limiter)
//--------------------------------------------------------------
RL::GeneratePoints::GeneratePoints( const size_t& generated_points, 
                                    const size_t& generated_points_test, 
                                    const size_t& column, 
                                    const double& limiter) :    Generate(generated_points, generated_points_test),
                                                                m_column(column),
                                                                m_gen(m_rd()),
                                                                m_random_point(-limiter, limiter){
    //--------------------------
}// end RL::GeneratePoints::GeneratePoints( const size_t& generated_points, const size_t& generated_points_test, const size_t& column, const double& limiter)
//--------------------------------------------------------------
std::vector<torch::Tensor> RL::GeneratePoints::get_input(void){
    //--------------------------
    return generate_value(this->get_generated_points(), m_column);
    //--------------------------
}// end torch::Tensor RL::GeneratePoints::get_input(void)
//--------------------------------------------------------------
std::vector<torch::Tensor> RL::GeneratePoints::get_test_input(void){
    //--------------------------
    return generate_value(this->get_generated_points_test(), m_column);
    //--------------------------
}// end std::vector<torch::Tensor> RL::GeneratePoints::get_test_input(void)
//--------------------------------------------------------------
torch::Tensor RL::GeneratePoints::get_output(const size_t& generated_points, const size_t& column){
    //--------------------------
    return generate_target(generated_points, column);
    //--------------------------
}// end torch::Tensor RL::GeneratePoints::get_output(const size_t& generated_points, const size_t& column)
//--------------------------------------------------------------
torch::Tensor RL::GeneratePoints::get_output(const size_t& generated_points, const size_t& points_size, const size_t& column){
    //--------------------------
    return generate_target(generated_points, points_size, column);
    //--------------------------
}// end torch::Tensor RL::GeneratePoints::get_output(const size_t& generated_points, const size_t& points_size, const size_t& column)
//--------------------------------------------------------------
std::vector<torch::Tensor> RL::GeneratePoints::data(const size_t& generated_points, const size_t& column){
    //--------------------------
    return generate_value(generated_points, column);
    //--------------------------
}// end torch::Tensor RL::GeneratePoints::data(const size_t& generated_points, const size_t& column)
//--------------------------------------------------------------
std::vector<torch::Tensor> RL::GeneratePoints::generate_value(const size_t& generated_points, const size_t& column){
    //--------------------------
    // https://stackoverflow.com/questions/66396651/what-is-the-most-efficient-way-of-converting-a-stdvectorstdtuple-to-a-to
    //--------------------------
    return Generate::generate_value<torch::Tensor>(generated_points, std::bind(&RL::GeneratePoints::inner_generation, this, std::placeholders::_1), column);
    //--------------------------
}// end std::vector<torch::Tensor> RL::GeneratePoints::generate_value(const size_t& generated_points, const size_t& column)
//--------------------------------------------------------------
torch::Tensor RL::GeneratePoints::inner_generation(const size_t& column){
    //--------------------------
    std::vector<double> _temp;
    _temp.reserve(column);
    //--------------------------
    std::vector<double> _output_data;
    _output_data.reserve(column-1);
    //--------------------------
    std::generate_n(std::execution::par, std::back_inserter(_temp), column-1, [this](){return m_random_point(m_gen);});
    //--------------------------
    std::generate_n(std::execution::par, std::back_inserter(_output_data), column-1, [this](){return m_random_point(m_gen);});
    //--------------------------
    _temp.push_back((std::pow((_output_data.at(0) - _temp.at(0)),2) + std::pow(( _output_data.at(1) - _temp.at(1)),2)));
    //--------------------------
    return torch::tensor(_temp).view({-1,static_cast<int64_t>(column)});
    //--------------------------
}// end torch::Tensor RL::GeneratePoints::inner_generation(const size_t& column)
//--------------------------------------------------------------
torch::Tensor RL::GeneratePoints::generate_target(const size_t& generated_points, const size_t& column){
    //--------------------------
    return torch::tensor(Generate::generate_value<double>(generated_points*column, [this]() {return m_random_point(m_gen);})).view({-1, static_cast<int64_t>(column)});
    //--------------------------
}// end torch::Tensor RL::GeneratePoints::generate_target(const size_t& generated_points, const size_t& column)
//--------------------------------------------------------------
torch::Tensor RL::GeneratePoints::generate_target(const size_t& generated_points, const size_t& points_size,const size_t& column){
    //--------------------------
    return torch::tensor(Generate::generate_value<double>(generated_points*column, [this]() {return m_random_point(m_gen);})).view({-1, static_cast<int64_t>(points_size), static_cast<int64_t>(column)});
    //--------------------------
}// end torch::Tensor RL::GeneratePoints::generate_target(const size_t& generated_points, const size_t& column)
//--------------------------------------------------------------