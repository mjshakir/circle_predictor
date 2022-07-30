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
torch::Tensor RLEnvironment::internal_cost_function(const torch::Tensor& predicted_value){
    return predicted_value - get_target();
}// end double RLEnvironment::internal_cost_function(const torch::Tensor& predicted_value)
//--------------------------------------------------------------