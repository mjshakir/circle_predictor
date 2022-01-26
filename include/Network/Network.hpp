#pragma once

#include <torch/torch.h>

struct Net : torch::nn::Module {
  //--------------------------------------------------------------
  public:
    //--------------------------
    Net(torch::Device& device):   m_device(device),
                                  fc1(torch::nn::LinearOptions(20, 128).bias(true)), 
                                  fc2(torch::nn::LinearOptions(128, 512).bias(true)), 
                                  fc3(torch::nn::LinearOptions(512, 1024).bias(true)), // 640
                                  fc4(torch::nn::LinearOptions(1024, 40).bias(true)){
      //--------------------------
      register_module("fc1", fc1);
      register_module("fc2", fc2);
      register_module("fc3", fc3);
      register_module("fc4", fc4);
      //--------------------------
    }// end Net(torch::Device& device)
    //--------------------------------------------------------------
    torch::Tensor forward(torch::Tensor& x){
      //--------------------------
      auto _results = torch::leaky_relu(fc1->forward(x), 5E-2);
      //--------------------------
      // auto _results = torch::relu(fc1->forward(x));
      //--------------------------
      _results = torch::relu(fc2->forward(_results));
      //--------------------------
      _results = torch::relu(fc3->forward(_results));
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
    torch::nn::Linear fc3;
    torch::nn::Linear fc4;
    //--------------------------------------------------------------
};
