#pragma once
//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------

class GenerateRL{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        GenerateRL() = delete;
        //--------------------------
        /**
         *  @brief A constructor. Generate both X and Y points. 
         *
         *  @tparam radius: circle raduis default value = 1.
         *  @tparam generated_points: How many points to generate
         *  @tparam center: points of the circle. 
         */
        GenerateRL(const size_t& generated_points = 60000, const double& limiter = 10.f);
        //--------------------------
        /**
         *  @brief Getter: the network input and target data set.  
         *
         *  @return 1) x_value: input data. 2)y_value: target data. 
         */
        torch::Tensor get_data(void);
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        torch::Tensor generate_value(void) const;
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        size_t m_generated_points;
        double m_limiter;
        torch::Tensor m_data;
        //--------------------------
    //--------------------------------------------------------------
};// end class Generate
//--------------------------------------------------------------