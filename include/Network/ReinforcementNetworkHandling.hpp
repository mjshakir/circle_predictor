#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------

template<typename Network>
class ReinforcementNetworkHandling{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        ReinforcementNetworkHandling() = delete;
        //--------------------------
        ReinforcementNetworkHandling(Network& model, const torch::Device& device): m_model(model), m_device(device){
            //--------------------------
        }// end ReinforcementNetworkHandling(Network& model, torch::Device& device)
        //--------------------------
        ReinforcementNetworkHandling(Network&& model, const torch::Device& device) : m_model(std::move(model)), m_device(device){
            //--------------------------
        }// end ReinforcementNetworkHandling(Network& model, torch::Device& device)
        //--------------------------
    //--------------------------------------------------------------
    protected:
        //--------------------------
    private:
        //--------------------------
        Network m_model; 
        torch::Device m_device;
        //--------------------------
    //--------------------------------------------------------------
};// end class ReinforcementNetworkHandling : protected NetworkHandling
//--------------------------------------------------------------