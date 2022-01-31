#pragma once

#include <torch/torch.h>

struct Net : torch::nn::Module {
  //--------------------------------------------------------------
  public:
    //--------------------------
    Net(torch::Device& device):   m_device(device),
                                  input_layer(torch::nn::LinearOptions(20, 128).bias(true)), 
                                  features(torch::nn::LinearOptions(128, 512).bias(true)), 
                                  features2(torch::nn::LinearOptions(532, 544).bias(true)), //1024
                                  output_layer(torch::nn::LinearOptions(1024, 40).bias(true)), //1536
                                  recurrent_layer(torch::nn::LSTMOptions(10, 80).num_layers(64).batch_first(true).bidirectional(true)){ // 128
      //--------------------------
      register_module("input_layer", input_layer);
      register_module("features", features);
      register_module("features2", features2);
      register_module("output_layer", output_layer);
      register_module("recurrent_layer", recurrent_layer);
      //--------------------------
    }// end Net(torch::Device& device)
    //--------------------------------------------------------------
    torch::Tensor forward(torch::Tensor& x){
      //--------------------------
      // auto x_linear = std::async(std::launch::async, [&](){return linear_layers(x);});
      auto x_linear = linear_layers(x);
      auto x_lstm = lstm_layers(x);
      //--------------------------
      auto out_results = torch::cat({x_linear, x_lstm});
      //--------------------------
      torch::dropout(out_results, /*p=*/0.5, /*training=*/is_training());
      //--------------------------
      return output_layer->forward(out_results);
      //--------------------------
      // torch::dropout(x_linear, /*p=*/0.5, /*training=*/is_training());
      //-------------------------
      // return output_layer->forward(x_linear);
      //--------------------------
    }// end torch::Tensor forward(torch::Tensor x)
    //--------------------------------------------------------------
  protected:
    //--------------------------------------------------------------
    torch::Tensor linear_layers(torch::Tensor& x){
      //--------------------------
      auto _results = torch::leaky_relu(input_layer->forward(x), 5E-2);
      //--------------------------
      // auto _results = torch::relu(input_layer->forward(x));
      //--------------------------
      _results = torch::relu(features->forward(_results));
      //--------------------------
      auto cat_results = torch::cat({_results, x});
      //--------------------------
      return torch::relu(features2->forward(cat_results));
      //-------------------------
    }// end torch::Tensor linear_layers(torch::Tensor& x)
    //--------------------------------------------------------------
    torch::Tensor lstm_layers(torch::Tensor& x){
      //--------------------------
      std::get<0>(_gates) = h0.to(m_device);
      std::get<1>(_gates) = c0.to(m_device);
      //--------------------------
      auto _input_lstm = x.view({-1, 2, 80}).to(m_device);
      //--------------------------
      auto x_lstm = recurrent_layer->forward(_input_lstm, _gates);
      //--------------------------
      h0 = std::get<0>(std::get<1>(x_lstm));
      c0 = std::get<1>(std::get<1>(x_lstm));
      //-------------------------
      return std::get<0>(x_lstm).reshape(-1);
    }// end torch::Tensor lstm_layers(torch::Tensor& x)
    //--------------------------------------------------------------
  private:
    //--------------------------------------------------------------
    torch::Device m_device;
    //--------------------------
    torch::nn::Linear input_layer;
    torch::nn::Linear features;
    torch::nn::Linear features2;
    torch::nn::Linear output_layer;
    //--------------------------
    torch::nn::LSTM recurrent_layer;
    //--------------------------
    torch::Tensor h0 = torch::from_blob(std::vector<float>(1*64*2, 0.0).data(), {64*2, 1, 80});
    torch::Tensor c0 = torch::from_blob(std::vector<float>(1*64*2, 0.0).data(), {64*2, 1, 80});
    //--------------------------
    std::tuple<torch::Tensor, torch::Tensor> _gates;
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> _input;
    //--------------------------------------------------------------
};
