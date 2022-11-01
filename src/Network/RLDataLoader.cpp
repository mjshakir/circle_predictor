//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Network/RLDataLoader.hpp"
//--------------------------------------------------------------
RLDataLoader::RLDataLoader(torch::Tensor&& input) : m_states_(std::move(input)), 
                                                    m_tensor(m_states_.data_ptr<float>()){
    //--------------------------
}// end RLDataLoader::RLDataLoader(const torch::Tensor& input, const torch::Tensor& target)
//--------------------------------------------------------------
torch::data::Example<> RLDataLoader::get(size_t index){
    // You may for example also read in a .csv file that stores locations
    // to your data and then read in the data at this step. Be creative.
    //--------------------------
    return m_states_[index];
    //--------------------------
}// end torch::data::Example<> RLDataLoader::get(size_t index)
//--------------------------------------------------------------
torch::optional<size_t> RLDataLoader::size() const{
    //--------------------------
    return m_tensor.size();
    //--------------------------
}// end torch::optional<size_t> RLDataLoader::size() const
//--------------------------------------------------------------