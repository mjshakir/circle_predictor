#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------

class RLNormalize{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        RLNormalize() = delete;
        //--------------------------
        /**
         *  @brief A constructor 
         *
         *  @tparam input: a torch tensor
         */
        RLNormalize(const std::vector<torch::Tensor>& input);
        //--------------------------
        /**
         *  @brief RLNormalize the input tensor from the constructor. This uses Min-max feature scaling.
         *
         *  @return A normalized a torch tensor
         */
        std::vector<torch::Tensor> normalization(void);
        //--------------------------
        /**
         *  @brief RLNormalize the input tensor from the constructor. This uses Min-max feature scaling.
         *
         *  @return A normalized a torch tensor
         */
        torch::Tensor normalization(const torch::Tensor& input);
        //--------------------------
        /**
         *  @brief  A static Normalizing function. This uses Min-max feature scaling. 
         *          Warning: cannot use unnormalization.
         * 
         *  @warning Cannot use unnormalization since you are using a static function. 
         *
         *  @return A normalized a torch tensor
         */
        static std::vector<torch::Tensor> normalization(const std::vector<torch::Tensor>& input);
        //--------------------------
        /**
         *  @brief  A static Normalizing function. This uses Min-max feature scaling. 
         *          Warning: cannot use unnormalization.
         * 
         *  @warning Cannot use unnormalization since you are using a static function. 
         *
         *  @return A normalized a torch tensor
         */
        static std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor> normalization_min_max(const std::vector<torch::Tensor>& input);
        //--------------------------
        /**
         *  @brief  Unnormalize the input tensor. This uses inverse Min-max feature scaling.
         * 
         *  @return A unnormalize a torch tensor
         */
        torch::Tensor unnormalization(const torch::Tensor& input);
        //--------------------------
        /**
         *  @brief  Unnormalize the input tensor. This uses inverse Min-max feature scaling.
         * 
         *  @return A unnormalize a torch tensor
         */
        static torch::Tensor unnormalization(const torch::Tensor& input, const torch::Tensor t_min, const torch::Tensor t_max);
        //--------------------------
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        std::vector<torch::Tensor> normalization_data(void);
        //--------------------------
        torch::Tensor normalization_data(const torch::Tensor& input);
        //--------------------------
        static std::vector<torch::Tensor> normalization_data(const std::vector<torch::Tensor>& input);
        //--------------------------
        static std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor>  normalization_min_max_data(const std::vector<torch::Tensor>& input);
        //--------------------------
        torch::Tensor unnormalization_data(const torch::Tensor& input);
        //--------------------------
        static torch::Tensor unnormalization_data(const torch::Tensor& input, const torch::Tensor t_min, const torch::Tensor t_max);
        //--------------------------
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        torch::Tensor m_max, m_min;
        //--------------------------
        std::vector<torch::Tensor> m_input;
        //--------------------------------------------------------------
        static std::tuple<torch::Tensor, torch::Tensor> find_min_max(std::vector<torch::Tensor> input);
        //--------------------------------------------------------------
    //--------------------------------------------------------------
};