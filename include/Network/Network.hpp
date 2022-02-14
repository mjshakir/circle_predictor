#pragma once

#include <torch/torch.h>
//--------------------------------------------------------------
struct Net : torch::nn::Module {
  //--------------------------------------------------------------
  public:
    //--------------------------
    Net() :   input_layer(torch::nn::LinearOptions(20, 128).bias(true)), 
              features(torch::nn::LinearOptions(128, 256).bias(true)), 
              features2(torch::nn::LinearOptions(276, 1024).bias(true)),
              output_layer(torch::nn::LinearOptions(1024, 40).bias(true)){
      //--------------------------
      register_module("input_layer", input_layer);
      register_module("features", features);
      register_module("features2", features2);
      register_module("output_layer", output_layer);
      //--------------------------
    }// end Net()
    //--------------------------------------------------------------
    torch::Tensor forward(torch::Tensor& x){
      //--------------------------
      auto x_linear = linear_layers(x);
      //--------------------------
      torch::dropout(x_linear, /*p=*/0.5, /*training=*/is_training());
      //-------------------------
      return output_layer->forward(x_linear);
      //--------------------------
    }// end torch::Tensor forward(torch::Tensor x)
    //--------------------------------------------------------------
  protected:
    //--------------------------------------------------------------
    torch::Tensor linear_layers(torch::Tensor& x){
      //--------------------------
      auto _results = torch::leaky_relu(input_layer->forward(x), 5E-2);
      //--------------------------
      _results = torch::relu(features->forward(_results));
      //--------------------------
      auto cat_results = torch::cat({_results, x});
      //--------------------------
      return torch::relu(features2->forward(cat_results));
      //-------------------------
    }// end torch::Tensor linear_layers(torch::Tensor& x)
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
struct LSTMNet : torch::nn::Module {
  //--------------------------------------------------------------
  public:
    //--------------------------
    LSTMNet(torch::Device& device): m_device(device),
                                    recurrent_layer(torch::nn::LSTMOptions(10, 128).num_layers(64).batch_first(true).bidirectional(true).dropout(0.5)),
                                    output_layer(torch::nn::LinearOptions(512, 40).bias(true)){
      //--------------------------
      register_module("recurrent_layer", recurrent_layer);
      register_module("output_layer", output_layer);
      //--------------------------
    }// end Net(torch::Device& device)
    //--------------------------------------------------------------
    torch::Tensor forward(torch::Tensor& x){
      //--------------------------
      x = lstm_layers(x);
      //--------------------------
      // torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
      //--------------------------
      return output_layer->forward(x.nan_to_num(1));
      //--------------------------
    }// end torch::Tensor forward(torch::Tensor x)
    //--------------------------------------------------------------
  protected:
    //--------------------------------------------------------------
    torch::Tensor lstm_layers(torch::Tensor& x){
      //--------------------------
      std::get<0>(_gates) = h0.to(m_device);
      std::get<1>(_gates) = c0.to(m_device);
      //--------------------------
      auto _input_lstm = x.view({-1, 2, 10}).to(m_device);
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
    torch::nn::LSTM recurrent_layer;
    torch::nn::Linear output_layer;
    //--------------------------
    torch::Tensor h0 = torch::from_blob(std::vector<float>(1*64*2, 0.0).data(), {64*2, 1, 128});
    torch::Tensor c0 = torch::from_blob(std::vector<float>(1*64*2, 0.0).data(), {64*2, 1, 128});
    //--------------------------
    std::tuple<torch::Tensor, torch::Tensor> _gates;
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> _input;
    //--------------------------------------------------------------
}; // end struct LSTMNet : torch::nn::Module
//--------------------------------------------------------------

