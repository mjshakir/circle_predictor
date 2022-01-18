#pragma once
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
class Generate{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        Generate() = delete;
        //--------------------------
        Generate(const double& radius = 1, const size_t& generated_points = 60000);
        //--------------------------
        Generate(const torch::Tensor& x_value, const double& radius = 1, const size_t& generated_points = 60000);
        //--------------------------
        torch::Tensor get_input(void); 
        //--------------------------
        torch::Tensor get_target(void);
        //--------------------------
        std::tuple<torch::Tensor, torch::Tensor> get_data(void);
        //--------------------------
        std::tuple<torch::Tensor, torch::Tensor> get_validation(void);
        //--------------------------
        double get_radius(void);
        //--------------------------------------------------------------
    
    protected:
        //--------------------------------------------------------------
        // Functions
        //--------------------------
        const std::tuple<torch::Tensor, torch::Tensor> generate_value(const double& radius);
        //--------------------------
        const std::tuple<torch::Tensor, torch::Tensor> generate_validation_value(const double& radius);
        //--------------------------
        const torch::Tensor generate_value(const torch::Tensor& x_value, const double& radius);
        //--------------------------------------------------------------
    private:
        //--------------------------
        // Variables
        //--------------------------
        double m_radius;
        //--------------------------
        size_t m_generated_points;
        //--------------------------
        torch::Tensor m_x_value, y_value;
        //--------------------------
        std::tuple<torch::Tensor, torch::Tensor> full_data, validation_data;
        //--------------------------------------------------------------
};