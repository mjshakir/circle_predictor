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
        RLGenerate(void) = delete;
        //--------------------------
        /**
         *  @brief A constructor. Generate both X, Y center point of the circle, and the radius. 
         *  it also generate test data that 20% of the generated_points
         *  @param generated_points: How many points to generate. @default: 60000
         *  @param column: helps design the structure of the output. @default: 2
         *  @param limiter: the maximum value the points can get for X, Y. @default: 10.0
         */
        RLGenerate(const size_t& generated_points = 60000, const size_t& column = 2, const double& limiter = 10.);
        //--------------------------
        /**
         *  @brief A constructor. Generate both X, Y center point of the circle, and the radius. 
         *  it also generate test data that 20% of the generated_points
         *  @param generated_points: How many points to generate. @default: 60000
         *  @param generated_points_test: How many points to test data is generated. @default: 10000
         *  @param column: helps design the structure of the output. @default: 2
         *  @param limiter: the maximum value the points can get for X, Y. @default: 10.0
         */
        RLGenerate( const size_t& generated_points = 60000,
                    const size_t& generated_points_test = 10000, 
                    const size_t& column = 2, 
                    const double& limiter = 10.);
        //--------------------------
        /**
         *  @brief Getter: The data generated.  
         *  
         *  @return td::vector<torch::Tensor>: Return the data X,Y, Radius.
         */
        std::vector<torch::Tensor> get_input(void);
        //--------------------------
        /**
         *  @brief Getter: The test data generated.  
         *  
         *  @return std::vector<torch::Tensor>: Return the data X,Y, Radius.
         */
        std::vector<torch::Tensor> get_test_input(void);
        //--------------------------
        /**
         *  @brief Getter: the target data set.
         *   
         *  @param generated_points: How many points to generate. @default: 60000
         *  @param column: helps design the structure of the output. @default: 2
         * 
         *  @return torch::Tensor: target data.
         */
        torch::Tensor get_output(const size_t& generated_points = 60000, const size_t& column = 2);
        //--------------------------
        /**
         * @brief Generated data.
         * 
         * @param generated_points: How many points to generate. @default: 60000
         * @param column: helps design the structure of the output. @default: 2
         * @return std::vector<torch::Tensor>: the data
         */
        std::vector<torch::Tensor> data(const size_t& generated_points = 60000, const size_t& column = 3);
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        std::vector<torch::Tensor> generate_value(const size_t& generated_points = 60000, const size_t& column = 2);
        //--------------------------
        std::vector<torch::Tensor> generate_input(const size_t& generated_points = 60000, const size_t& column = 2);
        //--------------------------
        torch::Tensor generate_target(const size_t& generated_points = 60000, const size_t& column = 2);
        //--------------------------
        torch::Tensor inner_generation(const size_t& column = 2);
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        double m_limiter;
        //--------------------------
        std::vector<torch::Tensor> m_data, m_data_test;
    //--------------------------------------------------------------
};// end class RLGenerate
//--------------------------------------------------------------