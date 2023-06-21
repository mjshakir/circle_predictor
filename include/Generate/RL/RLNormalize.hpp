#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
namespace RL {
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
             *  @param input: std::vector<torch::Tensor>: The data to be normalized. 
             */
            RLNormalize(const std::vector<torch::Tensor>& input);
            //--------------------------
            /**
             * @brief RLNormalize the input tensor from the constructor. This uses Min-max feature scaling.
             * 
             * @return std::vector<torch::Tensor>: A normalized data.
             */
            std::vector<torch::Tensor> normalization(void);
            //--------------------------
            /**
             * @brief RLNormalize the input tensor from the constructor. This uses Min-max feature scaling.
             * 
             * @param input: torch::Tensor: the input data in tensor format. 
             * @return torch::Tensor: A normalized data.
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
            static torch::Tensor unnormalization(const torch::Tensor& input, const torch::Tensor& t_min, const torch::Tensor& t_max);
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
            static torch::Tensor unnormalization_data(const torch::Tensor& input, const torch::Tensor& t_min, const torch::Tensor& t_max);
            //--------------------------------------------------------------
        private:
            //--------------------------------------------------------------
            std::vector<torch::Tensor> m_input;
            //--------------------------
            torch::Tensor m_input_tensor, m_min, m_max;
            //--------------------------------------------------------------
            std::tuple<torch::Tensor, torch::Tensor> find_min_max(const std::vector<torch::Tensor>& input);
            //--------------------------
            torch::Tensor find_min(const std::vector<torch::Tensor>& input);
            //--------------------------
            torch::Tensor find_max(const std::vector<torch::Tensor>& input);
            //--------------------------------------------------------------
        //--------------------------------------------------------------
    };// end class RLNormalize
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------