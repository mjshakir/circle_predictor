#pragma once

#include <torch/torch.h>


class DataLoader : public torch::data::Dataset<DataLoader>{

    public:
        //--------------------------
        // explicit DataLoader(const std::string& loc_states, const std::string& loc_labels);
        explicit DataLoader(const torch::Tensor& input, const torch::Tensor& target);
        //--------------------------
        torch::data::Example<> get(size_t index) override;
        //--------------------------
        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override;
        //--------------------------------------------------------------
    private:
        //--------------------------
        torch::Tensor m_states_, m_labels_;
        //--------------------------
        std::vector<double> m_tesnor;
        //--------------------------------------------------------------
};
//--------------------------------------------------------------