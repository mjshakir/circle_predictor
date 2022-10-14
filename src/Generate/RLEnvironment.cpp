//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/RLEnvironment.hpp"
//--------------------------------------------------------------
RLEnvironment::RLEnvironment(   const double& radius, 
                                const size_t& generated_points, 
                                const std::tuple<double, double>& center) : Generate(std::move(radius), std::move(generated_points), std::move(center)){
    //--------------------------
}// end RLEnvironment::RLEnvironment(const double& radius, const size_t& generated_points, const std::tuple<double, double>& center)
//--------------------------------------------------------------
const double RLEnvironment::internal_reward_function(const torch::Tensor& real_value, const torch::Tensor& predicted_value, const long double& tolerance){
    //--------------------------
    auto _difference = torch::abs(real_value - predicted_value) / ((real_value >= predicted_value) ? real_value : predicted_value);
    //--------------------------
    if(_difference > tolerance){
        return _difference*10;
    }
    //--------------------------
    return 100.f;
    //--------------------------
}// end double RLEnvironment::internal_reward_function(void)
//--------------------------------------------------------------