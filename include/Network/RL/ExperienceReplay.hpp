#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------

class ExperienceReplay{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        ExperienceReplay() = delete;
        //--------------------------
        ExperienceReplay(const size_t& capacity = 500);
        //--------------------------
        void push(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& reward, const bool& done);
        //--------------------------
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> sample(void);
        //--------------------------
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample(bool& done);
        //--------------------------
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample(const size_t& batch, bool& done);
        //--------------------------
        size_t size(void) const;
        //--------------------------------------------------------------
    protected:
        //--------------------------------------------------------------
        void push_data(const torch::Tensor& input, const torch::Tensor& next_input, const torch::Tensor& reward, const bool& done);
        //--------------------------
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool> sample_data(void);
        //--------------------------
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_data(bool& done);
        //--------------------------
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> sample_data(const size_t& batch, bool& done);
        //--------------------------
        size_t map_size(void) const;
        //--------------------------------------------------------------
    private:
        //--------------------------------------------------------------
        size_t m_capacity, m_position;
        //--------------------------
        std::unordered_map<size_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, bool>> m_memory;
        //--------------------------
    //--------------------------------------------------------------
};