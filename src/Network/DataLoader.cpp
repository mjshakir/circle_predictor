//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Network/DataLoader.hpp"

//--------------------------------------------------------------
DataLoader::DataLoader(torch::Tensor&& input, torch::Tensor&& target) : m_states_(std::move(input)), m_labels_(std::move(target)), 
                                                                        m_tensor(m_states_.data_ptr<float>(), m_states_.data_ptr<float>() + m_states_.numel()){
    //--------------------------
}// end DataLoader::DataLoader(const torch::Tensor& input, const torch::Tensor& target)
//--------------------------------------------------------------
torch::data::Example<> DataLoader::get(size_t index){
    // You may for example also read in a .csv file that stores locations
    // to your data and then read in the data at this step. Be creative.
    //--------------------------
    return {m_states_[index], m_labels_[index]};
    //--------------------------
}// end torch::data::Example<> DataLoader::get(size_t index)
//--------------------------------------------------------------
torch::optional<size_t> DataLoader::size() const{
    //--------------------------
    return m_tensor.size();
    //--------------------------
}// end torch::optional<size_t> DataLoader::size() const
//--------------------------------------------------------------