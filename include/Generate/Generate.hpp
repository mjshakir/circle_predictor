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
        /**
         *  @brief A constructor. Generate both X and Y points. 
         *
         *  @tparam radius: circle raduis default value = 1.
         *  @tparam generated_points: How many points to generate
         *  @tparam center: points of the circle. 
         */
        Generate(const double& radius = 1, const size_t& generated_points = 60000, const std::tuple<double, double>& center = {0,0});
        //--------------------------
        /**
         *  @brief A constructor. Generate Y points given an X point.  
         *
         *  @tparam x_value: the x point of the circle.
         *  @tparam radius: circle raduis default value = 1.
         *  @tparam generated_points: How many points to generate
         *  @tparam center: points of the circle. 
         */
        Generate(const torch::Tensor& x_value, const double& radius = 1, const size_t& generated_points = 60000, const std::tuple<double, double>& center = {0,0});
        //--------------------------
        /**
         *  @brief Getter: the network input data set.  
         *
         *  @return x_value: input data tensor. 
         */
        torch::Tensor get_input(void); 
        //--------------------------
        /**
         *  @brief Getter: the network target data set.  
         *
         *  @return y_value: target data tensor. 
         */
        torch::Tensor get_target(void);
        //--------------------------
        /**
         *  @brief Getter: the network input and target data set.  
         *
         *  @return 1) x_value: input data. 2)y_value: target data. 
         */
        std::tuple<torch::Tensor, torch::Tensor> get_data(void);
        //--------------------------
        /**
         *  @brief Getter: the validation input and target data set.  
         *
         *  @return 1) x_value: input data tensor. 2)y_value: target data tensor. 
         */
        std::tuple<torch::Tensor, torch::Tensor> get_validation(void);
        //--------------------------
        /**
         *  @brief Getter: the center of the circle.  
         *
         *  @return 1) x: point. 2) y: point. 
         */
        std::tuple<double, double> get_center(void);
        //--------------------------
        /**
         *  @brief Getter: the circle radius.  
         *
         *  @return 1) r: radius. 
         */
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
        //--------------------------
        const std::vector<double> generate_value(const std::vector<double>& x_value, const double& radius);
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        // Variables
        //--------------------------
        double m_radius;
        //--------------------------
        size_t m_generated_points;
        //--------------------------
        torch::Tensor m_x_value, y_value;
        //--------------------------
        std::tuple<double, double> m_center;
        //--------------------------
        std::tuple<torch::Tensor, torch::Tensor> full_data, validation_data;
        //--------------------------------------------------------------
};