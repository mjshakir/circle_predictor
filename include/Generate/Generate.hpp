#pragma once

#include <iostream>
#include <mutex>
#include <memory>
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
        Generate(const double& radius = 1, const uint16_t& generated_points = 60000);
        //--------------------------
        Generate(const torch::Tensor& x_value, const double& radius = 1, const uint16_t& generated_points = 60000);
        //--------------------------
        torch::Tensor get_x_value(void); 
        //--------------------------
        torch::Tensor get_y_value(void);
        //--------------------------
        GenerateDate get_data(void);
        //--------------------------
        double get_radius();
        //--------------------------------------------------------------
    
    protected:
        //--------------------------------------------------------------
        Generate(Generate &other) = delete;
        //--------------------------
        void operator=(const Generate &) = delete; 
        //--------------------------------------------------------------
    private:
        //--------------------------
        // Variables
        //--------------------------
        double m_radius;
        //--------------------------
        uint16_t m_generated_points;
        //--------------------------
        torch::Tensor m_x_value, y_value;
        //--------------------------
        GenerateDate full_data;
        //--------------------------
        // Functions
        //--------------------------
        const GenerateDate generate_value(const double& radius);
        //--------------------------
        const torch::Tensor generate_value(const torch::Tensor& x_value, const double& radius);
        //--------------------------------------------------------------
};