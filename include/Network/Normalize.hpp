#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------

class Normalize{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        Normalize() = delete;
        //--------------------------
        /**
         *  @brief A constructor 
         *
         *  @tparam input: a torch tensor
         */
        Normalize(const torch::Tensor& input);
        //--------------------------
        /**
         *  @brief A constructor 
         *
         *  @tparam input: a torch tensor
         */
        Normalize(const std::vector<torch::Tensor>& input);
        //--------------------------
        /**
         *  @brief Normalize the input tensor from the constructor. This uses Min-max feature scaling.
         *
         *  @return A normalized a torch tensor
         */
        torch::Tensor normalization(void);
        //--------------------------
        /**
         *  @brief Normalize the input tensor from the constructor. This uses Min-max feature scaling.
         *
         *  @return A normalized a torch tensor
         */
        std::vector<torch::Tensor> vnormalization(void);
        //--------------------------
        /**
         *  @brief Normalize the input tensor from the constructor. This uses Min-max feature scaling.
         *
         *  @return A normalized a torch tensor
         */
        torch::Tensor vnormalization(const torch::Tensor& input);
        //--------------------------
        /**
         *  @brief  A static Normalizing function. This uses Min-max feature scaling. 
         *          Warning: cannot use unnormalization.
         * 
         *  @warning Cannot use unnormalization since you are using a static function. 
         *
         *  @return A normalized a torch tensor
         */
        static torch::Tensor normalization(const torch::Tensor& input);
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
         *  @brief  Unnormalize the input tensor. This uses inverse Min-max feature scaling.
         * 
         *  @return A unnormalize a torch tensor
         */
        torch::Tensor unnormalization(const torch::Tensor& input);
        //--------------------------
        /**
         *  @brief  Template: unnormalize the input tensor. This uses Min-max feature scaling.
         * 
         *  @return Depends on the input type
         */
        template<typename T>
        T unnormalization_nonTensor(const T& input){
            //--------------------------
            return unnormalization_data_nonTensor(input);
            //--------------------------
        }// end T unnormalization_data_nontensor(const T& input)
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        torch::Tensor normalization_data(void);
        //--------------------------
        std::vector<torch::Tensor> normalization_vdata(void);
        //--------------------------
        torch::Tensor normalization_vdata(const torch::Tensor& input);
        //--------------------------
        static torch::Tensor normalization_data(const torch::Tensor& input);
        //--------------------------
        static std::vector<torch::Tensor> normalization_data(const std::vector<torch::Tensor>& input);
        //--------------------------
        torch::Tensor unnormalization_data(const torch::Tensor& input);
        //--------------------------
        template<typename T>
        T unnormalization_data_nonTensor(const T& input){
            //--------------------------
            return (input*(m_max.item<T>()-m_min.item<T>()))+m_min.item<T>();
            //--------------------------
        }// end T unnormalization_data_nontensor(const T& input)
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        torch::Tensor m_input, m_max, m_min;
        //--------------------------
        std::vector<torch::Tensor> m_vinput;
    //--------------------------------------------------------------
};
