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
         *  @brief Normalize the input tensor from the constructor. This uses Min-max feature scaling.
         *
         *  @return A normalized a torch tensor
         */
        torch::Tensor normalization(void);
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
        static torch::Tensor normalization_data(const torch::Tensor& input);
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
    //--------------------------------------------------------------
};
