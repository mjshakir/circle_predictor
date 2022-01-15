#include "Network/NetworkHandling.hpp"
//--------------------------------------------------------------
NetworkHandling::NetworkHandling(Net& model, torch::Device& device) : m_model(model), m_device(device){
    //--------------------------
}// end NetworkHandling::NetworkHandling(Net& model, torch::Device& device)
//--------------------------------------------------------------
void NetworkHandling::loss_display(const std::vector<float>& loss, const double& elements_sum){
    //--------------------------
    auto _max_element = std::max_element(std::execution::par_unseq, loss.begin(), loss.end());
    auto _min_element = std::min_element(std::execution::par_unseq, loss.begin(), loss.end());
    //--------------------------
    printf("\n-----------------Loss Sum[%f]---------Min[%ld] loss:[%f]---------Max:[%ld] loss[%f]-----------------\n", 
            elements_sum,
            std::distance(loss.begin(), _min_element), 
            *_min_element,  
            std::distance(loss.begin(), _max_element), 
            *_max_element);
    //--------------------------
}// end void NetworkHandling::loss_display(std::vector<float>, double elements_sum)
//--------------------------------------------------------------