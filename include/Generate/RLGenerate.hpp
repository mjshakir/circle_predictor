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
        RLGenerate(const size_t& generated_points = 60000, const double& limiter = 10.f);
        //--------------------------
        /**
         *  @brief Getter: the network input and target data set.  
         *
         *  @return 1) x_value: input data. 2)y_value: target data. 
         */
        std::vector<torch::Tensor> get_data(void);
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        std::vector<torch::Tensor> generate_value(const size_t& column = 3);
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        size_t m_generated_points;
        double m_limiter;
        //--------------------------
        std::vector<torch::Tensor> m_data;
        //--------------------------
    //--------------------------------------------------------------
};// end class Generate
//--------------------------------------------------------------