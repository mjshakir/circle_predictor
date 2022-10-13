//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/RLEnvironment.hpp"
//--------------------------------------------------------------
RLEnvironment::RLEnvironment(   const double& radius, 
                                const size_t& generated_points, 
                                const std::tuple<double, double>& center) : Generate(std::move(radius), std::move(generated_points), std::move(center)){
    //--------------------------
    [input, output] = get_data();
    m_input_data = input.accessor<float, 2>();
    m_output_data = output.accessor<float, 2>();
    //--------------------------
}// end RLEnvironment::RLEnvironment(const double& radius, const size_t& generated_points, const std::tuple<double, double>& center)
//--------------------------------------------------------------
double RLEnvironment::internal_cost_function(const double& input, const double& predicted_value){
    return predicted_value - get_target();
}// end double RLEnvironment::internal_cost_function(const torch::Tensor& predicted_value)
//--------------------------------------------------------------
double RLEnvironment::internal_reward_function(void){
    
}// end double RLEnvironment::internal_reward_function(void)
//--------------------------------------------------------------