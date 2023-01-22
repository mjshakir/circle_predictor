#pragma once
//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------

class RLGenerate{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        RLGenerate() = delete;
        //--------------------------
        /**
         *  @brief A constructor. Generate both X and Y points. 
         *
         *  @tparam radius: circle raduis default value = 1.
         *  @tparam generated_points: How many points to generate
         *  @tparam center: points of the circle. 
         */
        RLGenerate(const size_t& generated_points = 60000, const size_t& column = 2, const double& limiter = 10.f);
        //--------------------------
        /**
         *  @brief Getter: the network input and target data set.  
         *
         *  @return 1) x_value: input data. 2)y_value: target data. 
         */
        std::vector<torch::Tensor> get_input(void);
        //--------------------------
        /**
         *  @brief Getter: the network input and target data set.  
         *
         *  @return 1) x_value: input data. 2)y_value: target data. 
         */
        torch::Tensor get_output(const size_t& generated_points = 60000, const size_t& column = 2);
        //--------------------------
        /**
         *  @brief Getter: the network input and target data set.  
         *
         *  @return 1) x_value: input data. 2)y_value: target data. 
         */
        torch::Tensor get_test_output(const size_t& generated_points = 60000, const size_t& column = 2);
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        std::vector<torch::Tensor> generate_value(const size_t& generated_points = 60000, const size_t& column = 2);
        //--------------------------
        std::vector<torch::Tensor> generate_input(const size_t& generated_points = 60000, const size_t& column = 2);
        //--------------------------
        torch::Tensor generate_target(const size_t& generated_points = 60000, const size_t& column = 2);
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        size_t m_generated_points, m_column;
        double m_limiter;
        //--------------------------
        std::vector<torch::Tensor> m_data;
        //--------------------------
    //--------------------------------------------------------------
};
// end class RLGenerate
//--------------------------------------------------------------