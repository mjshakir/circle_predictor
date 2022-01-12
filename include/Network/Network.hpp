#pragma once

#include <torch/torch.h>

struct Net : torch::nn::Module {
  Net(torch::Device& device):   m_device(device),
                                fc1(torch::nn::LinearOptions(20, 15).bias(true)), 
                                fc2(torch::nn::LinearOptions(15, 10).bias(true)), 
                                middel(torch::nn::LSTMOptions(5, 15).num_layers(20).batch_first(true).bidirectional(true)),
                                fc3(torch::nn::LinearOptions(60, 25).bias(true)), 
                                fc4(torch::nn::LinearOptions(25, 20).bias(true)){
    //--------------------------
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("middel", middel);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
    //--------------------------
  }// end Net(torch::Device& device)
  //--------------------------------------------------------------
  torch::Tensor forward(torch::Tensor& x){
    //--------------------------
    auto _results = torch::leaky_relu(fc1->forward(x));
    //--------------------------
    _results = torch::relu(fc2->forward(_results));
    //--------------------------
    std::get<0>(_gates) = h0.to(m_device);
    std::get<1>(_gates) = c0.to(m_device);
    //--------------------------
    auto _input_lstm = _results.view({-1, 2, 5}).to(m_device);
    auto x_lstm = middel->forward(_input_lstm, _gates);
    //--------------------------
    h0 = std::get<0>(std::get<1>(x_lstm));
    c0 = std::get<1>(std::get<1>(x_lstm));
    //-------------------------
    _results = torch::relu(fc3->forward(std::get<0>(x_lstm).view(-1)));
    //-------------------------
    _results = torch::dropout(_results, /*p=*/0.5, /*training=*/is_training());
    //--------------------------
    return fc4->forward(_results);
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
    torch::Tensor h0 = torch::from_blob(std::vector<float>(1*20*2, 0.0).data(), {40, 1, 15});
    torch::Tensor c0 = torch::from_blob(std::vector<float>(1*20*2, 0.0).data(), {40, 1, 15});
    //--------------------------
    std::tuple<torch::Tensor, torch::Tensor> _gates;
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> _input;
    //--------------------------------------------------------------
};
