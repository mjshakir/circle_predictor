//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Network/Networks.hpp"
//--------------------------------------------------------------
// Net struct 
//--------------------------------------------------------------
Net::Net(const uint64_t& batch_size) :  input_layer(torch::nn::LinearOptions(batch_size, 128).bias(true)), 
                                        features(torch::nn::LinearOptions(128, 256).bias(true)), 
                                        features2(torch::nn::LinearOptions((256+batch_size), 1024).bias(true)),
                                        output_layer(torch::nn::LinearOptions(1024, (2*batch_size)).bias(true)){
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
// RLNet struct 
//--------------------------------------------------------------
RLNet::RLNet(const uint64_t& batch_size, const uint64_t& output_size) : input_layer(torch::nn::LinearOptions(batch_size, 128).bias(true)), 
                                                                        features(torch::nn::LinearOptions(128, 256).bias(true)), 
                                                                        features2(torch::nn::LinearOptions(256, 512).bias(true)),
                                                                        output_layer(torch::nn::LinearOptions(512, output_size).bias(true)){
    //--------------------------
    register_module("input_layer", input_layer);
    register_module("features", features);
    register_module("features2", features2);
    register_module("output_layer", output_layer);
    //--------------------------
}// end RLNet()
//--------------------------------------------------------------
torch::Tensor RLNet::linear_layers(const torch::Tensor& x){
    //--------------------------
    auto _results = torch::leaky_relu(input_layer->forward(x), 5E-2);
    //--------------------------
    // std::cout << "[1]" << " results: " << _results.sizes() << std::endl;
    //--------------------------
    _results = torch::relu(features->forward(_results));
    //--------------------------
    // std::cout << "[2]" << " results: " << _results.sizes() << std::endl;
    //--------------------------
    return torch::relu(features2->forward(_results));
    //-------------------------
}// end torch::Tensor Net::linear_layers(torch::Tensor& x)
//--------------------------------------------------------------
torch::Tensor RLNet::forward(const torch::Tensor& x){
    //--------------------------
    auto _results = linear_layers(x);
    //--------------------------
    // std::cout << "x: " << x.sizes() << std::endl;
    //--------------------------
    torch::dropout(_results, /*p=*/0.5, /*training=*/is_training());
    //-------------------------
    // return torch::transpose(output_layer->forward(x).view({2,-1}), 0, 1);
    //--------------------------
    return output_layer->forward(_results);
    //--------------------------
}// end torch::Tensor Net::forward(torch::Tensor x)
//--------------------------------------------------------------
// RLNet struct 
//--------------------------------------------------------------
DuelNet::DuelNet(const uint64_t& batch_size, const uint64_t& output_size) : input_layer(torch::nn::LinearOptions(batch_size, 128).bias(true)), 
                                                                            features(torch::nn::LinearOptions(128, 256).bias(true)), 
                                                                            features2(torch::nn::LinearOptions(256, 512).bias(true)),
                                                                            value_layer(torch::nn::LinearOptions(512, 1).bias(true)),
                                                                            advantage_layer(torch::nn::LinearOptions(512, output_size).bias(true)){
    //--------------------------
    register_module("input_layer", input_layer);
    register_module("features", features);
    register_module("features2", features2);
    register_module("value_layer", value_layer);
    register_module("advantage_layer", advantage_layer);
    //--------------------------
}// end RLNet()
//--------------------------------------------------------------
torch::Tensor DuelNet::linear_layers(const torch::Tensor& x){
    //--------------------------
    auto _results = torch::leaky_relu(input_layer->forward(x), 5E-2);
    //--------------------------
    _results = torch::relu(features->forward(_results));
    //--------------------------
    return torch::relu(features2->forward(_results));
    //-------------------------
}// end torch::Tensor DuelNet::linear_layers(torch::Tensor& x)
//--------------------------------------------------------------
torch::Tensor DuelNet::forward(const torch::Tensor& x){
    //--------------------------
    auto _results = linear_layers(x);
    //--------------------------
    torch::dropout(_results, /*p=*/0.5, /*training=*/is_training());
    //-------------------------
    auto value = value_layer->forward(_results);
    //--------------------------
    auto advantage = advantage_layer->forward(_results);
    //--------------------------
    return (value + (advantage - advantage.mean()));
    //--------------------------
}// end torch::Tensor Net::forward(torch::Tensor x)
//--------------------------------------------------------------
// RLNetLSTM struct 
//--------------------------------------------------------------
RLNetLSTM::RLNetLSTM(const std::tuple<uint64_t, uint64_t>& input_size, const uint64_t& output_size, const torch::Device& device, const bool& inject) : 
            m_device(device),
            m_input_size(input_size),
            m_output_size(output_size),
            m_inject(inject),
            m_gates({torch::rand({static_cast<int64_t>(output_size* 2), static_cast<int64_t>(std::get<1>(input_size)), static_cast<int64_t>(std::get<1>(input_size))}).to(device),
                    torch::rand({static_cast<int64_t>(output_size* 2), static_cast<int64_t>(std::get<1>(input_size)), static_cast<int64_t>(std::get<1>(input_size))}).to(device)}),
            recurrent_layer(torch::nn::LSTMOptions(std::get<0>(input_size), std::get<1>(input_size)).num_layers(output_size).bias(true).batch_first(true).bidirectional(true).dropout(0.5)),
            input_layer(torch::nn::LinearOptions(std::get<1>(input_size)*output_size, 32).bias(true)), 
            features((inject) ? torch::nn::LinearOptions((32 + std::get<0>(input_size)), 64).bias(true) : torch::nn::LinearOptions(32, 64).bias(true)), 
            features2(torch::nn::LinearOptions(64, 128).bias(true)),
            output_layer(torch::nn::LinearOptions(128, output_size).bias(true)){
    //--------------------------
    register_module("recurrent_layer", recurrent_layer);
    //--------------------------
    register_module("input_layer", input_layer);
    register_module("features", features);
    register_module("features2", features2);
    register_module("output_layer", output_layer);
    //--------------------------
}// end RLNet()
//--------------------------------------------------------------
torch::Tensor RLNetLSTM::lstm_layers(const torch::Tensor& x){
    //--------------------------
    torch::Tensor x_lstm;
    std::tie(x_lstm, m_gates) = recurrent_layer->forward(x.view({static_cast<int64_t>(std::get<1>(m_input_size)), -1, static_cast<int64_t>(std::get<0>(m_input_size))}), m_gates);
    //--------------------------
    return x_lstm;
    //-------------------------
}// end torch::Tensor LSTMNet::lstm_layers(const torch::Tensor& x)
//--------------------------------------------------------------
torch::Tensor RLNetLSTM::linear_layers(const torch::Tensor& x){
    //--------------------------
    auto _results = torch::leaky_relu(input_layer->forward(x), 5E-2);
    //--------------------------
    _results = torch::relu(features->forward(_results));
    //--------------------------
    return torch::relu(features2->forward(_results));
    //-------------------------
}// end torch::Tensor Net::linear_layers(torch::Tensor& x)
//--------------------------------------------------------------
torch::Tensor RLNetLSTM::linear_layers(const torch::Tensor& input, const torch::Tensor& x){
    //--------------------------
    auto _results = torch::leaky_relu(input_layer->forward(x), 5E-2);
    //--------------------------
    _results = torch::relu(features->forward(torch::cat({_results.view({-1, 32}), input}, 1)));
    //--------------------------
    return torch::relu(features2->forward(_results));
    //-------------------------
}// end torch::Tensor Net::linear_layers(torch::Tensor& x)
//--------------------------------------------------------------
torch::Tensor RLNetLSTM::forward(const torch::Tensor& x){
    //--------------------------
    auto _result = lstm_layers(x);
    //--------------------------
    if(m_inject){
        //--------------------------
        _result = linear_layers(x, _result);
        //--------------------------
    }// end if(m_inject)
    else{
        //--------------------------
        _result = linear_layers(_result);
        //--------------------------
    }// end else
    //--------------------------
    torch::dropout(_result, /*p=*/0.5, /*training=*/is_training());
    //-------------------------
    return output_layer->forward(_result).view({-1, static_cast<int64_t>(m_output_size)});
    //--------------------------
}// end torch::Tensor Net::forward(torch::Tensor x)
//--------------------------------------------------------------
// LSTMNet struct 
//--------------------------------------------------------------
LSTMNet::LSTMNet(const torch::Device& device):  m_device(device),
                                                recurrent_layer(torch::nn::LSTMOptions(10, 10).num_layers(20).batch_first(false).bidirectional(true).dropout(0.5)){
    //--------------------------
    register_module("recurrent_layer", recurrent_layer);
    //--------------------------
    _gates = {h0.to(m_device), c0.to(m_device)};
    //--------------------------
}// end Net(torch::Device& device)
//--------------------------------------------------------------
torch::Tensor LSTMNet::forward(torch::Tensor& x){
    //--------------------------
    return lstm_layers(x);
    //--------------------------
}// end torch::Tensor LSTMNet::forward(const torch::Tensor& x)
//--------------------------------------------------------------
torch::Tensor LSTMNet::lstm_layers(torch::Tensor& x){
    //--------------------------
    x = x.view({-1, 1, 10}).to(m_device);
    //--------------------------
    auto x_lstm = recurrent_layer->forward(x, _gates);
    //--------------------------
    _gates = {std::get<0>(std::get<1>(x_lstm)), std::get<1>(std::get<1>(x_lstm))};
    //-------------------------
    return std::get<0>(x_lstm).view({20, -1});
    //-------------------------
}// end torch::Tensor LSTMNet::lstm_layers(const torch::Tensor& x)
//--------------------------------------------------------------