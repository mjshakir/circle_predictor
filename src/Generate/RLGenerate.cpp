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
    // std::vector<std::tuple<double, std::tuple<double, double>>> _data;
    // _data.reserve(m_generated_points);
    std::vector<torch::Tensor> _data;
    _data.reserve(m_generated_points);
    //--------------------------
    std::vector<double> _temp;
    _temp.reserve(3);
    //--------------------------
    for (size_t i = 0; i < m_generated_points; ++i){
        //--------------------------
        // std::cout << "tensor generation: " << std::endl;
        // _data.emplace_back(torch::transpose(torch::cat({torch::tensor(uniform_angle(re)), torch::tensor(uniform_angle(re)), torch::tensor(uniform_angle(re))}).view({3,-1}), 0, 1));
        // auto _radius = uniform_angle(re);
        // auto _center_x = uniform_angle(re);
        // auto _center_y = uniform_angle(re);
        // auto _temp = torch::cat({torch::tensor(uniform_angle(re)), torch::tensor(uniform_angle(re)), torch::tensor(uniform_angle(re))});
        
        for (size_t i = 0; i < column; ++i){ // end batch
            //--------------------------
            _temp.push_back(uniform_angle(re));
            //--------------------------
        }// end for (size_t i = 0; i < 20; ++i)
        //--------------------------
        // _data.push_back(torch::transpose(torch::cat({torch::tensor(uniform_angle(re)), torch::tensor(uniform_angle(re)), torch::tensor(uniform_angle(re))}).view({-1,3}), 0, 1));
        //--------------------------
        auto _data_temp = torch::tensor(_temp);
        _temp.clear();
        //--------------------------
        _data.push_back(_data_temp.view({-1,3}));
        //--------------------------
    }// end for (size_t i = 0; i < m_generated_points; i++)
    //--------------------------
    return _data;
    //--------------------------
}// end torch::Tensor RLGenerate::generate_value(void)
//--------------------------------------------------------------