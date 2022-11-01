#pragma once
//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------

class RLDataLoader : public torch::data::Dataset<RLDataLoader>{
    //--------------------------------------------------------------
    public:
        //--------------------------
        // explicit DataLoader(const std::string& loc_states, const std::string& loc_labels);
        explicit RLDataLoader(torch::Tensor&& input);
        //--------------------------
        torch::data::Example<> get(size_t index) override;
        //--------------------------
        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override;
        //--------------------------------------------------------------
    private:
        //--------------------------
        torch::Tensor m_states_;
        //--------------------------
        std::vector<double> m_tensor;
        //--------------------------------------------------------------
};// end class RLDataLoader : public torch::data::Dataset<RLDataLoader>
//--------------------------------------------------------------