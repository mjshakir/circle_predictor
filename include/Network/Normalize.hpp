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
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        torch::Tensor normalization_data(void);
        //--------------------------
        static torch::Tensor normalization_data(const torch::Tensor& input);
        //--------------------------
        torch::Tensor unnormalization_data(const torch::Tensor& input);
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        torch::Tensor m_input, m_max, m_min;
    //--------------------------------------------------------------
};
