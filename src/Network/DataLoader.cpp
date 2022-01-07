#include "Network/DataLoader.hpp"

//--------------------------------------------------------------
DataLoader::DataLoader(const torch::Tensor& input, const torch::Tensor& target) :   m_states_(input), m_labels_(target), 
                                                                                    m_tesnor(input.data_ptr<float>(), input.data_ptr<float>() + input.numel()){
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
    return m_tesnor.size();
    //--------------------------
}// end torch::optional<size_t> DataLoader::size() const
//--------------------------------------------------------------