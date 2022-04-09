#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// User defined library  
//--------------------------------------------------------------
#include "NetworkHandling.hpp"
//--------------------------------------------------------------

class ReinforcementNetworkHandling : protected NetworkHandling{
    //--------------------------------------------------------------
    public:
        //--------------------------------------------------------------
        NetworkHandling() = delete;
        //--------------------------
        ReinforcementNetworkHandling(Network& model, const torch::Device& device): NetworkHandling(model, device){
            //--------------------------
        }// end NetworkHandling(Network& model, torch::Device& device)
        //--------------------------
    //--------------------------------------------------------------
};// end class ReinforcementNetworkHandling : protected NetworkHandling
//--------------------------------------------------------------