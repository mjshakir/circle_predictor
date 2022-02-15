#include "Network/Network.hpp"

//--------------------------------------------------------------
// Net struct 
//--------------------------------------------------------------
Net::Net() : input_layer(torch::nn::LinearOptions(20, 128).bias(true)), 
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
torch::Tensor Net::forward(torch::Tensor& x){
    //--------------------------
    x = linear_layers(x);
    //--------------------------
    torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
    //-------------------------
    return torch::transpose(output_layer->forward(x).view({2,-1}), 0, 1);
    //--------------------------
}// end torch::Tensor Net::forward(torch::Tensor x)
//--------------------------------------------------------------
torch::Tensor Net::linear_layers(const torch::Tensor& x){
    //--------------------------
    auto _results = torch::leaky_relu(input_layer->forward(x), 5E-2);
    //--------------------------
    _results = torch::relu(features->forward(_results));
    //--------------------------
    auto cat_results = torch::cat({_results, x});
    //--------------------------
    return torch::relu(features2->forward(cat_results));
    //-------------------------
}// end torch::Tensor Net::linear_layers(torch::Tensor& x)
//--------------------------------------------------------------
// LSTMNet struct 
//--------------------------------------------------------------
LSTMNet::LSTMNet(const torch::Device& device):  m_device(device),
                                                recurrent_layer(torch::nn::LSTMOptions(10, 10).num_layers(20).batch_first(false).bidirectional(true).dropout(0.5)){
    //--------------------------
    register_module("recurrent_layer", recurrent_layer);
    //--------------------------
    std::get<0>(_gates) = h0.to(m_device);
    std::get<1>(_gates) = c0.to(m_device);
    //--------------------------
}// end Net(torch::Device& device)
//--------------------------------------------------------------
torch::Tensor LSTMNet::forward(const torch::Tensor& x){
    //--------------------------
    return lstm_layers(x);
    //--------------------------
}// end torch::Tensor LSTMNet::forward(const torch::Tensor& x)
//--------------------------------------------------------------
torch::Tensor LSTMNet::lstm_layers(const torch::Tensor& x){
    //--------------------------
    auto _input_lstm = x.view({-1, 1, 10}).to(m_device);
    //--------------------------
    auto x_lstm = recurrent_layer->forward(_input_lstm, _gates);
    //--------------------------
    _gates = {std::get<0>(std::get<1>(x_lstm)), std::get<1>(std::get<1>(x_lstm))};
    //-------------------------
    return std::get<0>(x_lstm).view({20, -1});
    //-------------------------
}// end torch::Tensor LSTMNet::lstm_layers(const torch::Tensor& x)
//--------------------------------------------------------------