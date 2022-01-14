#pragma once

#include <iostream>
//--------------------------------------------------------------
#include "SharedLibrary/SharedLibrary.hpp"
//--------------------------------------------------------------
class Generate{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        Generate() = delete;
        //--------------------------
        // ~Generate();
        //--------------------------
        Generate(const double& radius = 1, const size_t& generated_points = 60000);
        //--------------------------
        Generate(const torch::Tensor& x_value, const double& radius = 1, const size_t& generated_points = 60000);
        //--------------------------
        torch::Tensor get_input(void); 
        //--------------------------
        torch::Tensor get_target(void);
        //--------------------------
        GenerateDate get_data(void);
        //--------------------------
        GenerateDate get_test(void);
        //--------------------------
        double get_radius(void);
        //--------------------------------------------------------------
    
    protected:
        //--------------------------------------------------------------
        // Functions
        //--------------------------
        const GenerateDate generate_value(const double& radius);
        //--------------------------
        const GenerateDate generate_test_value(const double& radius);
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
        GenerateDate full_data, test_data;
        //--------------------------------------------------------------
};