#pragma once

#include <torch/torch.h>

struct Net : torch::nn::Module {
  Net():  fc1(torch::nn::LinearOptions(20, 15).bias(true)), 
          fc2(torch::nn::LinearOptions(15, 10).bias(true)), 
          middel(torch::nn::LSTMOptions(5, 5).num_layers(5).batch_first(true).bidirectional(true)),
          fc3(torch::nn::LinearOptions(10, 10).bias(true)), 
          fc4(torch::nn::LinearOptions(10, 20).bias(true)){
    //--------------------------
    std::get<0>(_gates) = h0;
    std::get<1>(_gates) = c0;
    //--------------------------
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("middel", middel);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
    //--------------------------
  }// end Net()
  //--------------------------------------------------------------
  torch::Tensor forward(torch::Tensor x){
    //--------------------------
    x = torch::leaky_relu(fc1->forward(x));
    //--------------------------
    x = torch::relu(fc2->forward(x));
    //--------------------------
    std::get<0>(_gates) = h0;
    std::get<1>(_gates) = c0;
    //--------------------------
    auto _input_lstm = x.view({-1, 2, 5});
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
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::LSTM middel;
    torch::nn::Linear fc3;
    torch::nn::Linear fc4;
    //--------------------------
    torch::Tensor h0 = torch::from_blob(std::vector<float>(1*5*2, 0.0).data(), {10, 1, 5});
    torch::Tensor c0 = torch::from_blob(std::vector<float>(1*5*2, 0.0).data(), {10, 1, 5});
    //--------------------------
    std::tuple<torch::Tensor, torch::Tensor> _gates;
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> _input;
    //--------------------------------------------------------------
};
