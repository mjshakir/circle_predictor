//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/RL/GeneratePoints.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
//-------------------
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    // short form to get the previous even: x &= ~1;
    GeneratePoints::GeneratePoints( const size_t& generated_points, 
                                    const size_t& column, 
                                    const double& limiter) :    Generate(generated_points),
                                                                m_generated_points(this->get_generated_points()), 
                                                                m_generated_points_test(this->get_generated_points_test()),
                                                                m_column(column),
                                                                m_limiter(limiter){
        //--------------------------
    }// end GeneratePoints::GeneratePoints(const size_t& generated_points, const size_t& column, const double& limiter)
    //--------------------------------------------------------------
    GeneratePoints::GeneratePoints( const size_t& generated_points, 
                                    const size_t& generated_points_test, 
                                    const size_t& column, 
                                    const double& limiter) : Generate(generated_points, generated_points_test),
                                                             m_generated_points(this->get_generated_points()), 
                                                             m_generated_points_test(this->get_generated_points_test()),
                                                             m_column(column),
                                                             m_limiter(limiter){
        //--------------------------
    }// end GeneratePoints::GeneratePoints( const size_t& generated_points, const size_t& generated_points_test, const size_t& column, const double& limiter)
    //--------------------------------------------------------------
    std::vector<torch::Tensor> GeneratePoints::get_input(void){
        //--------------------------
        return generate_value(m_generated_points, m_column);
        //--------------------------
    }// end torch::Tensor GeneratePoints::get_input(void)
    //--------------------------------------------------------------
    std::vector<torch::Tensor> GeneratePoints::get_test_input(void){
        //--------------------------
        return generate_value(m_generated_points_test, m_column);
        //--------------------------
    }// end std::vector<torch::Tensor> GeneratePoints::get_test_input(void)
    //--------------------------------------------------------------
    torch::Tensor GeneratePoints::get_output(const size_t& generated_points, const size_t& column){
        //--------------------------
        return generate_target(generated_points, column);
        //--------------------------
    }// end std::vector<torch::Tensor> GeneratePoints::get_output(const size_t& generated_points, const size_t& column)
    //--------------------------------------------------------------
    std::vector<torch::Tensor> GeneratePoints::data(const size_t& generated_points, const size_t& column){
        //--------------------------
        return generate_value(generated_points, column);
        //--------------------------
    }// end torch::Tensor GeneratePoints::data(const size_t& generated_points, const size_t& column)
    //--------------------------------------------------------------
    std::vector<torch::Tensor> GeneratePoints::generate_value(const size_t& generated_points, const size_t& column){
        //--------------------------
        // https://stackoverflow.com/questions/66396651/what-is-the-most-efficient-way-of-converting-a-stdvectorstdtuple-to-a-to
        //--------------------------
        return Generate::generate_value<torch::Tensor>(generated_points, std::bind(&GeneratePoints::inner_generation, this, std::placeholders::_1), column);
        //--------------------------
    }// end std::vector<torch::Tensor> GeneratePoints::generate_value(const size_t& generated_points, const size_t& column)
    //--------------------------------------------------------------
    torch::Tensor GeneratePoints::inner_generation(const size_t& column){
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
        std::generate_n(std::execution::par, std::back_inserter(_temp), column-1, [&uniform_angle, &gen](){return uniform_angle(gen);});
        //--------------------------
        std::generate_n(std::execution::par, std::back_inserter(_output_data), column-1, [&uniform_angle, &gen](){return uniform_angle(gen);});
        //--------------------------
        _temp.push_back((std::pow((_output_data.at(0) - _temp.at(0)),2) + std::pow(( _output_data.at(1) - _temp.at(1)),2)));
        //--------------------------
        return torch::tensor(_temp).view({-1,static_cast<int64_t>(column)});
        //--------------------------
    }// end torch::Tensor GeneratePoints::inner_generation(const size_t& column)
    //--------------------------------------------------------------
    torch::Tensor GeneratePoints::generate_target(const size_t& generated_points, const size_t& column){
        //--------------------------
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
        std::uniform_real_distribution<double> uniform_angle(-m_limiter, m_limiter);
        //--------------------------
        std::vector<double> _data;
        _data.reserve(generated_points*column);
        //--------------------------
        std::generate_n(std::execution::par, std::back_inserter(_data), generated_points*column, [&uniform_angle, &gen]() {return uniform_angle(gen);});
        //--------------------------
        return torch::tensor(_data).view({-1, static_cast<int64_t>(column)});
        //--------------------------
    }// end torch::Tensor GeneratePoints::generate_target(const size_t& generated_points, const size_t& column)
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------