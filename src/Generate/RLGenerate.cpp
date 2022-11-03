//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/RLGenerate.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
//--------------------------------------------------------------
RLGenerate::RLGenerate(const size_t& generated_points, const double& limiter) : m_generated_points(generated_points), m_limiter(limiter) {
    //--------------------------
    m_data = generate_value();
    //--------------------------
}// end RLGenerate::RLGenerate(const size_t& generated_points)
//--------------------------------------------------------------
std::vector<torch::Tensor> RLGenerate::get_data(void){
    //--------------------------
    return m_data;
    //--------------------------
}// end torch::Tensor RLGenerate::get_data(void)
//--------------------------------------------------------------
std::vector<torch::Tensor> RLGenerate::generate_value(const size_t& column){
    //--------------------------
    // https://stackoverflow.com/questions/66396651/what-is-the-most-efficient-way-of-converting-a-stdvectorstdtuple-to-a-to
    //--------------------------
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd(
    std::uniform_real_distribution<double> uniform_angle(-m_limiter, m_limiter);
    std::default_random_engine re;
    //--------------------------
    std::vector<torch::Tensor> _data;
    _data.reserve(m_generated_points);
    //--------------------------
    std::vector<double> _temp;
    _temp.reserve(3);
    //--------------------------
    for (size_t i = 0; i < m_generated_points; ++i){
        //--------------------------
        for (size_t i = 0; i < column; ++i){ // end batch
            //--------------------------
            _temp.push_back(uniform_angle(re));
            //--------------------------
        }// end for (size_t i = 0; i < column; ++i)
        //--------------------------
        // _data.push_back(torch::transpose(torch::cat({torch::tensor(uniform_angle(re)), torch::tensor(uniform_angle(re)), torch::tensor(uniform_angle(re))}).view({-1,3}), 0, 1));
        //--------------------------
        _data.push_back(torch::tensor(_temp).view({-1,3}));
        //--------------------------
        _temp.clear();
        //--------------------------
    }// end for (size_t i = 0; i < m_generated_points; ++i)
    //--------------------------
    return _data;
    //--------------------------
}// end torch::Tensor RLGenerate::generate_value(void)
//--------------------------------------------------------------