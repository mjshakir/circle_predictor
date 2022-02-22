#include <torch/torch.h>

class Normalize{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        Normalize() = delete;
        //--------------------------
        Normalize(const torch::Tensor& input);
        //--------------------------
        torch::Tensor normalization(void);
        //--------------------------
        static torch::Tensor normalization(const torch::Tensor& input);
        //--------------------------
        torch::Tensor unnormalization(const torch::Tensor& input);
        //--------------------------
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
