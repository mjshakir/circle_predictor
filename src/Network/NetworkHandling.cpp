#include "Network/NetworkHandling.hpp"
//--------------------------------------------------------------
NetworkHandling::NetworkHandling(Net& model, torch::Device& device) : m_model(model), m_device(device){
    //--------------------------
}// end NetworkHandling::NetworkHandling(Net& model, torch::Device& device)
//--------------------------------------------------------------