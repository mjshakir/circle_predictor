#pragma once

#include <torch/torch.h>

struct Net : torch::nn::Module {
  Net(torch::Device& device):   m_device(device),
                                fc1(torch::nn::LinearOptions(20, 15).bias(true)), 
                                fc2(torch::nn::LinearOptions(15, 10).bias(true)), 
                                middel(torch::nn::LSTMOptions(5, 5).num_layers(64).batch_first(true).bidirectional(true)),
                                fc3(torch::nn::LinearOptions(10, 10).bias(true)), 
                                fc4(torch::nn::LinearOptions(10, 20).bias(true)){
    //--------------------------
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("middel", middel);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
    //--------------------------
  }// end Net(torch::Device& device)
  //--------------------------------------------------------------
  torch::Tensor forward(torch::Tensor x){
    //--------------------------
    x = torch::leaky_relu(fc1->forward(x));
    //--------------------------
    x = torch::relu(fc2->forward(x));
    //--------------------------
    std::get<0>(_gates) = h0.to(m_device);
    std::get<1>(_gates) = c0.to(m_device);
    //--------------------------
    auto _input_lstm = x.view({-1, 2, 5}).to(m_device);
    auto x_lstm = middel->forward(_input_lstm, _gates);
    //--------------------------
    h0 = std::get<0>(std::get<1>(x_lstm));
    c0 = std::get<1>(std::get<1>(x_lstm));
    //-------------------------
    x = torch::relu(fc3->forward(std::get<0>(x_lstm)));
    //-------------------------
    x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    //--------------------------
    return fc4->forward(x);
    //--------------------------
  }// end torch::Tensor forward(torch::Tensor x)
  //--------------------------------------------------------------
  private:
    //--------------------------------------------------------------
    torch::Device m_device;
    //--------------------------
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::LSTM middel;
    torch::nn::Linear fc3;
    torch::nn::Linear fc4;
    //--------------------------
    torch::Tensor h0 = torch::from_blob(std::vector<float>(1*64*2, 0.0).data(), {128, 1, 5});
    torch::Tensor c0 = torch::from_blob(std::vector<float>(1*64*2, 0.0).data(), {128, 1, 5});
    //--------------------------
    std::tuple<torch::Tensor, torch::Tensor> _gates;
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> _input;
    //--------------------------------------------------------------
};
