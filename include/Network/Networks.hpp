#pragma once
//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>

//--------------------------------------------------------------
struct Net : torch::nn::Module {
  //--------------------------------------------------------------
  public:
    //--------------------------
    Net(const uint64_t& batch_size = 20); 
    //--------------------------
    Net(const uint64_t& batch_size, const uint64_t& output_size);
    //--------------------------
    torch::Tensor forward(torch::Tensor& x);
    //--------------------------------------------------------------
  protected:
    //--------------------------------------------------------------
    torch::Tensor linear_layers(const torch::Tensor& x);
    //--------------------------------------------------------------
  private:
    //--------------------------------------------------------------
    torch::nn::Linear input_layer;
    torch::nn::Linear features;
    torch::nn::Linear features2;
    torch::nn::Linear output_layer;
    //--------------------------------------------------------------
}; // end struct Net : torch::nn::Module
//--------------------------------------------------------------
struct RLNet : torch::nn::Module {
  //--------------------------------------------------------------
  public:
    //--------------------------
    RLNet(const uint64_t& batch_size, const uint64_t& output_size);
    //--------------------------
    RLNet(const uint64_t& batch_size, const uint64_t& points_size, const uint64_t& output_size);
    //--------------------------
    torch::Tensor forward(const torch::Tensor& x);
    //--------------------------------------------------------------
  protected:
    //--------------------------------------------------------------
    torch::Tensor linear_layers(const torch::Tensor& x);
    //--------------------------------------------------------------
  private:
    //--------------------------------------------------------------
    torch::nn::Linear input_layer;
    torch::nn::Linear features;
    torch::nn::Linear features2;
    torch::nn::Linear output_layer;
    //--------------------------
    uint64_t m_output_size, m_points_size;
    //--------------------------------------------------------------
}; // end struct Net : torch::nn::Module
//--------------------------------------------------------------
struct DuelNet : torch::nn::Module {
  //--------------------------------------------------------------
  public:
    //--------------------------
    DuelNet(const uint64_t& batch_size, const uint64_t& output_size);
    //--------------------------
    DuelNet(const uint64_t& batch_size, const uint64_t& points_size, const uint64_t& output_size);
    //--------------------------
    torch::Tensor forward(const torch::Tensor& x);
    //--------------------------------------------------------------
  protected:
    //--------------------------------------------------------------
    torch::Tensor linear_layers(const torch::Tensor& x);
    //--------------------------------------------------------------
  private:
    //--------------------------------------------------------------
    torch::nn::Linear input_layer;
    torch::nn::Linear features;
    torch::nn::Linear features2;
    torch::nn::Linear value_layer;
    torch::nn::Linear advantage_layer;
    //--------------------------
    uint64_t m_output_size, m_points_size;
    //--------------------------------------------------------------
}; // end struct Net : torch::nn::Module
//--------------------------------------------------------------
struct RLNetLSTM : torch::nn::Module {
  //--------------------------------------------------------------
  public:
    //--------------------------
    RLNetLSTM(const std::tuple<uint64_t, uint64_t>& input_size,
              const uint64_t& output_size = 20,
              const torch::Device& device = torch::kCPU,
              const bool& inject = false);
    //--------------------------
    torch::Tensor forward(const torch::Tensor& x);
    //--------------------------------------------------------------
  protected:
    //--------------------------------------------------------------
    torch::Tensor lstm_layers(const torch::Tensor& x);
    //--------------------------------------------------------------
    torch::Tensor linear_layers(const torch::Tensor& x);
    //--------------------------------------------------------------
    torch::Tensor linear_layers(const torch::Tensor& input, const torch::Tensor& x);
    //--------------------------------------------------------------
  private:
    //--------------------------------------------------------------
    torch::Device m_device;
    //--------------------------
    std::tuple<uint64_t, uint64_t> m_input_size;
    //--------------------------
    size_t m_output_size;
    //--------------------------
    bool m_inject;
    //--------------------------
    std::tuple<torch::Tensor, torch::Tensor> m_gates;
    //--------------------------
    torch::nn::LSTM recurrent_layer;
    //--------------------------
    torch::nn::Linear input_layer;
    torch::nn::Linear features;
    torch::nn::Linear features2;
    torch::nn::Linear output_layer;
    //--------------------------
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> _input;
    //--------------------------------------------------------------
}; // end struct Net : torch::nn::Module
//--------------------------------------------------------------
struct LSTMNet : torch::nn::Module {
  //--------------------------------------------------------------
  public:
    //--------------------------
    LSTMNet(const torch::Device& device);
    //--------------------------------------------------------------
    torch::Tensor forward(torch::Tensor& x);
    //--------------------------------------------------------------
  protected:
    //--------------------------------------------------------------
    torch::Tensor lstm_layers(torch::Tensor& x);
    //--------------------------------------------------------------
  private:
    //--------------------------------------------------------------
    torch::Device m_device;
    //--------------------------
    torch::nn::LSTM recurrent_layer;
    //--------------------------
    torch::Tensor h0 = torch::from_blob(std::vector<float>(1*20*2, 0.0).data(), {20*2, 1, 10});
    torch::Tensor c0 = torch::from_blob(std::vector<float>(1*20*2, 0.0).data(), {20*2, 1, 10});
    //--------------------------
    std::tuple<torch::Tensor, torch::Tensor> _gates;
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> _input;
    //--------------------------------------------------------------
}; // end struct LSTMNet : torch::nn::Module
//--------------------------------------------------------------

